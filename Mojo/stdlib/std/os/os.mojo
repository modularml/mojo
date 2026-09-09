# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Provides functions to access operating-system dependent functionality, including
file system operations.

You can import a method from the `os` package. For example:

```mojo
from std.os import listdir
```
"""

from std._plugin import CurrentPlugin
from std.collections import Array, List
from std.collections.string.string_span import _unsafe_strlen
from std.format.tstring import TString
from std.io import FileDescriptor
from std.ffi import c_char, c_int, external_call, get_errno
from std.reflection import SourceLocation, call_location
from std._gpu import thread_idx, block_idx
from std.sys import CompilationTarget, is_gpu, is_apple_gpu

from .path import isdir, split, exists
from .pathlike import PathLike as stdPathLike

# TODO move this to a more accurate location once nt/posix like modules are in stdlib
comptime sep = "/"
"""The path separator for the current platform."""


# ===----------------------------------------------------------------------=== #
# SEEK Constants
# ===----------------------------------------------------------------------=== #


comptime SEEK_SET: UInt8 = 0
"""Seek from the beginning of the file."""
comptime SEEK_CUR: UInt8 = 1
"""Seek from the current position."""
comptime SEEK_END: UInt8 = 2
"""Seek from the end of the file."""


# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


struct _dirent_linux(Copyable):
    comptime MAX_NAME_SIZE = 256
    var d_ino: Int64
    """File serial number."""
    var d_off: Int64
    """Seek offset value."""
    var d_reclen: Int16
    """Length of the record."""
    var d_type: Int8
    """Type of file."""
    var name: Array[c_char, Self.MAX_NAME_SIZE]
    """Name of entry."""


struct _dirent_macos(Copyable):
    comptime MAX_NAME_SIZE = 1024
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
    var name: Array[c_char, Self.MAX_NAME_SIZE]
    """Name of entry."""


struct _DirHandle:
    """Handle to an open directory descriptor opened via opendir."""

    var _handle: OpaquePointer[MutUntrackedOrigin]

    def __init__(out self, var path: String) raises:
        """Construct the _DirHandle using the path provided.

        Args:
          path: The path to open.
        """
        if not isdir(path):
            raise Error("the directory '", path, "' does not exist")

        var handle = external_call[
            "opendir", OptionalPointer[NoneType, UntrackedOrigin[mut=True]]
        ](path.as_c_string_slice())

        if not handle:
            var err = get_errno()
            raise Error(
                "unable to open the directory '",
                path,
                "'",
                " Err: ",
                String(err),
            )

        self._handle = handle.value()

    def __deinit__(deinit self):
        """Closes the handle opened via popen."""
        _ = external_call["closedir", Int32](self._handle)

    def list(self) -> List[String]:
        """Reads all the data from the handle.

        Returns:
          A string containing the output of running the command.
        """

        comptime if CompilationTarget.is_linux():
            return self._list_linux()
        else:
            return self._list_macos()

    def _list_linux(self) -> List[String]:
        """Reads all the data from the handle.

        Returns:
            A string containing the output of running the command.
        """
        var res = List[String]()

        while True:
            var ep = external_call[
                "readdir", OptionalPointer[_dirent_linux, MutUntrackedOrigin]
            ](self._handle)
            if not ep:
                break
            ref name = ep.unsafe_value().unsafe_take_pointee().name
            var name_ptr = name.unsafe_ptr().unsafe_bitcast[Byte]()
            var name_str = StringSlice[origin_of(name)](
                unsafe_from_utf8=Span[Byte, origin_of(name)](
                    unsafe_ptr=name_ptr,
                    length=Int(
                        _unsafe_strlen(name_ptr, _dirent_linux.MAX_NAME_SIZE)
                    ),
                )
            )
            if name_str == "." or name_str == "..":
                continue
            res.append(String(name_str))

        return res^

    def _list_macos(self) -> List[String]:
        """Reads all the data from the handle.

        Returns:
            A string containing the output of running the command.
        """
        var res = List[String]()

        while True:
            var ep = external_call[
                "readdir", OptionalPointer[_dirent_macos, MutUntrackedOrigin]
            ](self._handle)
            if not ep:
                break
            ref name = ep.unsafe_value().unsafe_take_pointee().name
            var name_ptr = name.unsafe_ptr().unsafe_bitcast[Byte]()
            var name_str = StringSlice[origin_of(name)](
                unsafe_from_utf8=Span[Byte, origin_of(name)](
                    unsafe_ptr=name_ptr,
                    length=Int(
                        _unsafe_strlen(name_ptr, _dirent_macos.MAX_NAME_SIZE)
                    ),
                )
            )
            if name_str == "." or name_str == "..":
                continue
            res.append(String(name_str))

        return res^


# ===----------------------------------------------------------------------=== #
# getuid
# ===----------------------------------------------------------------------=== #
def getuid() -> Int:
    """Retrieve the user ID of the calling process.

    Returns:
        The user ID of the calling process.

    Constraints:
        This function is constrained to run on Linux or macOS operating systems only.
    """
    return Int(external_call["getuid", UInt32]())


# ===----------------------------------------------------------------------=== #
# listdir
# ===----------------------------------------------------------------------=== #


def listdir[PathLike: stdPathLike](path: PathLike) raises -> List[String]:
    """Gets the list of entries contained in the path provided.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.

    Returns:
      Returns the list of entries in the path provided.

    Raises:
        If the operation fails.
    """
    var dir = _DirHandle(path.__fspath__())
    return dir.list()


# ===----------------------------------------------------------------------=== #
# abort
# ===----------------------------------------------------------------------=== #


@always_inline
def _abort_base() -> Never:
    __mlir_op.`llvm.intr.trap`()

    # We need to satisfy the noreturn checker.
    while True:
        pass


@always_inline
def abort() -> Never:
    """Terminates execution, using a target dependent trap instruction if
    available.
    """

    # Plugin hook may longjmp
    # if so, the trap below is dead.
    CurrentPlugin.abort_fn()

    # If no hook, if hook fails, or if hook longjmps,
    # fall through to base impl.
    _abort_base()


@no_inline
def _abort_report[
    *, prefix: StaticString
](loc: SourceLocation, message: Some[Writable]):
    """Prints an abort message.

    Kept out of line so the wide `print` call below is emitted once per message
    type rather than inlined into every `abort` call site.
    """

    comptime if is_apple_gpu():
        # FIXME: Remove after MOCO-3697 is fixed.
        pass
    elif is_gpu():
        # On GPU, gate the print to a single thread to avoid flooding the
        # printf buffer with identical messages from thousands of threads.
        if (
            thread_idx.x == 0
            and thread_idx.y == 0
            and thread_idx.z == 0
            and block_idx.x == 0
            and block_idx.y == 0
            and block_idx.z == 0
        ):
            print(
                prefix,
                " ",
                loc,
                ": block: [",
                block_idx.x,
                ",",
                block_idx.y,
                ",",
                block_idx.z,
                "] thread: [",
                thread_idx.x,
                ",",
                thread_idx.y,
                ",",
                thread_idx.z,
                "]: ",
                message,
                sep="",
                flush=True,
            )
    else:
        print(
            prefix,
            StaticString(" "),
            loc,
            StaticString(": "),
            message,
            sep="",
            flush=True,
        )


@always_inline
def _abort_impl[
    *, prefix: StaticString
](
    message: Some[Writable],
    *,
    location: Optional[SourceLocation] = {},
) -> Never:
    var loc = location.or_else(call_location[inline_count=2]())

    # Apple GPU has no print path at all (see `_abort_report`), so its body
    # folds to nothing.
    comptime if not is_apple_gpu():
        _abort_report[prefix=prefix](loc, message)
    abort()


@always_inline
def abort[
    *, prefix: StaticString = "ABORT:"
](message: String, *, location: Optional[SourceLocation] = {}) -> Never:
    """Calls a target dependent trap instruction if available.

    Parameters:
        prefix: A static string prefix to include before the message.

    Args:
        message: The message to include when aborting.
        location: The optional source location to include.
    """
    _abort_impl[prefix=prefix](message, location=location)


@always_inline
def abort[
    *, prefix: StaticString = "ABORT:"
](message: TString, *, location: Optional[SourceLocation] = {}) -> Never:
    """Calls a target dependent trap instruction if available.

    Parameters:
        prefix: A static string prefix to include before the message.

    Args:
        message: The t-string message to include when aborting.
        location: The optional source location to include.
    """
    _abort_impl[prefix=prefix](message, location=location)


# ===----------------------------------------------------------------------=== #
# remove/unlink
# ===----------------------------------------------------------------------=== #
def remove[PathLike: stdPathLike](path: PathLike) raises:
    """Removes the specified file.

    If the path is a directory or it can not be deleted, an error is raised.
    Absolute and relative paths are allowed, relative paths are resolved from cwd.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the file.


    Raises:
        If the operation fails.
    """
    var fspath = path.__fspath__()
    var error = external_call["unlink", Int32](fspath.as_c_string_slice())

    if error != 0:
        var err = get_errno()
        raise Error("Can not remove file: ", fspath, " Err: ", String(err))


def unlink[PathLike: stdPathLike](path: PathLike) raises:
    """Removes the specified file.

    If the path is a directory or it can not be deleted, an error is raised.
    Absolute and relative paths are allowed, relative paths are resolved from cwd.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the file.


    Raises:
        If the operation fails.
    """
    remove(path.__fspath__())


# ===----------------------------------------------------------------------=== #
# symlink
# ===----------------------------------------------------------------------=== #


def symlink[
    TargetType: stdPathLike, LinkType: stdPathLike
](target: TargetType, linkpath: LinkType) raises:
    """Creates a symlink.

    If linkpath already exists it will not be overwritten.
    See `symlink(2)`

    Parameters:
        TargetType: The path type of the link target.
        LinkType: The path type of the link.

    Args:
        target: The target of the symbolic link.
        linkpath: The path of the symbolic link to create.

    Raises:
        If the operation fails.
    """
    var target_fspath = target.__fspath__()
    var linkpath_fspath = linkpath.__fspath__()

    var error = external_call["symlink", c_int](
        target_fspath.as_c_string_slice(),
        linkpath_fspath.as_c_string_slice(),
    )

    if error != 0:
        var err = get_errno()
        raise Error(
            "Can not create symlink from ",
            linkpath_fspath,
            " to ",
            target_fspath,
            " Err: ",
            String(err),
        )


# ===----------------------------------------------------------------------=== #
# link
# ===----------------------------------------------------------------------=== #


def link[
    OldType: stdPathLike, NewType: stdPathLike
](oldpath: OldType, newpath: NewType) raises:
    """Creates a new hard-link to an existing file.

    Parameters:
        OldType: The path type of the existing file.
        NewType: The path type of the file to create.

    Args:
        oldpath: The existing file.
        newpath: The new file.

    Raises:
        If the operation fails.
    """
    var oldpath_fspath = oldpath.__fspath__()
    var newpath_fspath = newpath.__fspath__()

    var error = external_call["link", Int32](
        oldpath_fspath.as_c_string_slice(),
        newpath_fspath.as_c_string_slice(),
    )

    if error != 0:
        var err = get_errno()
        raise Error(
            "Can not create link from ",
            newpath_fspath,
            " to ",
            oldpath_fspath,
            " Err: ",
            String(err),
        )


# ===----------------------------------------------------------------------=== #
# mkdir/rmdir
# ===----------------------------------------------------------------------=== #


def mkdir[PathLike: stdPathLike](path: PathLike, mode: Int = 0o777) raises:
    """Creates a directory at the specified path.

    If the directory can not be created an error is raised.
    Absolute and relative paths are allowed, relative paths are resolved from cwd.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.
      mode: The mode to create the directory with.

    Raises:
        If the operation fails.
    """

    var fspath = path.__fspath__()
    var error = external_call["mkdir", Int32](fspath.as_c_string_slice(), mode)
    if error != 0:
        var err = get_errno()
        raise Error("Can not create directory: ", fspath, " Err: ", String(err))


def makedirs[
    PathLike: stdPathLike
](path: PathLike, mode: Int = 0o777, exist_ok: Bool = False) raises -> None:
    """Creates a specified leaf directory along with any necessary intermediate
    directories that don't already exist.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.
      mode: The mode to create the directory with.
      exist_ok: Ignore error if `True` and path exists (default `False`).

    Raises:
        If the operation fails.
    """
    var head, tail = split(path)
    if not tail:
        head, tail = split(head)
    if head and tail and not exists(head):
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
            raise Error(
                e,
                "\nset `makedirs(path, exist_ok=True)` to allow existing dirs",
            )
        if not isdir(path):
            raise Error("path not created: ", path.__fspath__(), "\n", e)


def rmdir[PathLike: stdPathLike](path: PathLike) raises:
    """Removes the specified directory.

    If the path is not a directory or it can not be deleted, an error is raised.
    Absolute and relative paths are allowed, relative paths are resolved from cwd.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.

    Raises:
        If the operation fails.
    """
    var fspath = path.__fspath__()
    var error = external_call["rmdir", Int32](fspath.as_c_string_slice())
    if error != 0:
        var err = get_errno()
        raise Error("Can not remove directory: ", fspath, " Err: ", String(err))


def removedirs[PathLike: stdPathLike](path: PathLike) raises -> None:
    """Removes a leaf directory and all empty intermediate ones.

    Directories corresponding to rightmost path segments will be pruned away
    until either the whole path is consumed or an error occurs. Errors during
    this latter phase are ignored, which occur when a directory was not empty.

    Parameters:
      PathLike: A type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.

    Raises:
        If the operation fails.
    """
    rmdir(path)
    var head, tail = split(path)
    if not tail:
        head, tail = split(head)
    while head and tail:
        try:
            rmdir(head)
        except:
            break
        head, tail = split(head)


# ===----------------------------------------------------------------------=== #
# isatty
# ===----------------------------------------------------------------------=== #


def isatty(fd: Int) -> Bool:
    """Checks whether a file descriptor refers to a terminal.

    Returns `True` if the file descriptor `fd` is open and connected to a
    tty(-like) device, otherwise `False`. On GPUs, the function always returns
    `False`.

    Args:
        fd: A file descriptor.

    Returns:
        `True` if `fd` is connected to a terminal, `False` otherwise.

    Examples:
        ```mojo
        from std.os import isatty

        # Check if stdout (fd=1) is a terminal
        if isatty(1):
            print("Running in a terminal")
        else:
            print("Output is redirected")
        ```
    """

    return FileDescriptor(fd).isatty()


# ===----------------------------------------------------------------------=== #
# chdir
# ===----------------------------------------------------------------------=== #


def chdir[PathLike: stdPathLike](path: PathLike) raises:
    """Changes the current working directory.

    Parameters:
        PathLike: A type conforming to the os.PathLike trait.

    Args:
        path: The path to the new working directory.

    Raises:
        If the operation fails.
    """
    var fspath = path.__fspath__()
    var error = external_call["chdir", Int32](fspath.as_c_string_slice())
    if error != 0:
        var err = get_errno()
        raise Error("chdir failed: ", fspath, " Err: ", String(err))
