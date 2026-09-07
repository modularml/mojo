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
"""Temporary file and directory creation.

This module provides a safe API for creating temporary files and directories.
It includes helpers for creating temporary directories (`mkdtemp`),
temporary files (`NamedTemporaryFile`), querying the default temporary
directory (`gettempdir`), and a context manager for managing the lifetime of
temporary directories (`TemporaryDirectory`).

To use these features, import the desired functions or classes from this module.

```mojo
# Import from the tempfile module
from std.tempfile import gettempdir
```
"""

import std.os
from std.format._utils import _FlushingWriteBuffer
from std.pathlib import Path
from std.sys import CompilationTarget
from std.random import random_ui64

from std.collections import Span

comptime TMP_MAX = 10_000
"""Maximum number of attempts when generating unique temporary file names."""


def _get_random_name(size: Int = 8) -> String:
    comptime characters = StaticString("abcdefghijklmnopqrstuvwxyz0123456789_")
    var name = String(capacity_bytes=size)
    for _ in range(size):
        var rand_index = Int(
            random_ui64(0, UInt64(characters.byte_length() - 1))
        )
        name += characters[byte=rand_index]
    return name^


def _candidate_tempdir_list() -> List[String]:
    """Generate a list of candidate temporary directories which
    _get_default_tempdir will try."""

    var dirlist = List[String]()
    var possible_env_vars: List[StaticString] = ["TMPDIR", "TEMP", "TMP"]
    var dirname: String

    # First, try the environment.
    for env_var in possible_env_vars:
        if dirname := std.os.getenv(String(env_var)):
            dirlist.append(dirname^)

    # Failing that, try OS-specific locations.
    dirlist.extend(["/tmp", "/var/tmp", "/usr/tmp"])

    # As a last resort, the current directory if possible,
    # os.path.getcwd() could raise
    try:
        dirlist.append(String(Path()))
    except:
        pass

    return dirlist^


def _get_default_tempdir() raises -> String:
    """Calculate the default directory to use for temporary files.

    We determine whether or not a candidate temp dir is usable by
    trying to create and write to a file in that directory.  If this
    is successful, the test file is deleted. To prevent denial of
    service, the name of the test file must be randomized."""

    var dirlist = _candidate_tempdir_list()

    for dir_name in dirlist:
        if not std.os.path.isdir(dir_name):
            continue
        if _try_to_create_file(dir_name):
            return dir_name

    raise Error("No usable temporary directory found")


def _try_to_create_file(dir: StringSlice) -> Bool:
    for _ in range(TMP_MAX):
        var name = _get_random_name()
        # TODO use os.join when it exists
        var filename = Path(dir) / name

        # prevent overwriting existing file
        if std.os.path.exists(filename):
            continue

        # verify that we have writing access in the target directory
        try:
            with FileHandle(String(filename), "w"):
                pass
            std.os.remove(filename)
            return True
        except:
            if std.os.path.exists(filename):
                try:
                    std.os.remove(filename)
                except:
                    pass
            return False

    return False


def gettempdir() -> Optional[String]:
    """Return the default directory to use for temporary files.

    Returns:
        The name of the default temporary directory.

    Example:

    ```mojo
    from std.tempfile import gettempdir

    print(gettempdir())
    ```
    """
    # TODO In python _get_default_tempdir is called exactly once so that the default
    # tmp dir is the same throughout the program execution/
    # Since there is no global scope in mojo yet, this is not possible for now.
    try:
        return _get_default_tempdir()
    except:
        return None


def mkdtemp(
    suffix: String = "", prefix: String = "tmp", dir: Optional[String] = None
) raises -> String:
    """Create a temporary directory.

    Caller is responsible for deleting the directory when done with it.

    Args:
        suffix: Suffix to use for the directory name.
        prefix: Prefix to use for the directory name.
        dir: Directory in which the directory will be created.

    Returns:
        The name of the created directory.

    Raises:
        If the directory can not be created.

    Example:

    ```mojo
    from std.tempfile import mkdtemp
    from std.os import rmdir

    var temp_dir = mkdtemp()
    print(temp_dir)
    rmdir(temp_dir)
    ```
    """
    var final_dir = Path(dir.value()) if dir else Path(_get_default_tempdir())

    for _ in range(TMP_MAX):
        var dir_name = final_dir / (prefix + _get_random_name() + suffix)
        if std.os.path.exists(dir_name):
            continue
        try:
            std.os.mkdir(dir_name, mode=0o700)
            # TODO for now this name could be relative,
            # python implementation expands the path,
            # but several functions are not yet implemented in mojo
            # i.e. abspath, normpath
            return String(dir_name)
        except:
            continue
    raise Error("Failed to create temporary file")


# TODO use shutil.rmtree (or equivalent) when it exists
def _rmtree(path: String, ignore_errors: Bool = False) raises:
    """Removes the specified directory and all its contents.

    If the path is a symbolic link, an error is raised. If ignore_errors is
    True, errors resulting from failed removals will be ignored. Absolute and
    relative paths are allowed, relative paths are resolved from cwd.

    Args:
        path: The path to the directory.
        ignore_errors: Whether to ignore errors.

    Raises:
        If the operation fails.

    Example:

    ```mojo
    from std.tempfile import mkdtemp
    from std.tempfile.tempfile import _rmtree

    var dir_path = mkdtemp()

    # Add files and directories to dir_path

    _rmtree(dir_path)
    ```
    """
    if std.os.path.islink(path):
        raise Error("`path`can not be a symbolic link: ", path)

    for file_or_dir in std.os.listdir(path):
        var curr_path = std.os.path.join(path, file_or_dir)
        if std.os.path.isfile(curr_path):
            try:
                std.os.remove(curr_path)
            except e:
                if not ignore_errors:
                    raise e^
            continue
        if std.os.path.isdir(curr_path):
            try:
                _rmtree(curr_path, ignore_errors)
            except e:
                if ignore_errors:
                    continue
                raise e^
    try:
        std.os.rmdir(path)
    except e:
        if not ignore_errors:
            raise e^


struct TemporaryDirectory:
    """Temporary directory that cleans up automatically.

    Creates a directory that's deleted when the context exits:
    ```mojo
    from std.tempfile import TemporaryDirectory
    import std.os

    def main() raises:
        var temp_path: String
        with TemporaryDirectory() as tmpdir:
            temp_path = tmpdir
            print(std.os.path.exists(tmpdir))  # True
            # Use tmpdir for temporary work

        print(std.os.path.exists(temp_path))  # False - cleaned up
    ```

    The directory and all its contents are removed on exit, even if an error occurs.
    Set `ignore_cleanup_errors=True` to suppress cleanup failures.
    """

    var name: String
    """The name of the temporary directory."""
    var _ignore_cleanup_errors: Bool
    """Whether to ignore cleanup errors."""

    def __init__(
        out self,
        suffix: String = "",
        prefix: String = "tmp",
        dir: Optional[String] = None,
        ignore_cleanup_errors: Bool = False,
    ) raises:
        """Create a temporary directory.

        Can be used as a context manager. When used as a context manager,
        the directory is removed when the context manager exits.

        Args:
            suffix: Suffix to use for the directory name.
            prefix: Prefix to use for the directory name.
            dir: Directory in which the directory will be created.
            ignore_cleanup_errors: Whether to ignore cleanup errors.

        Raises:
            If the operation fails.
        """
        self._ignore_cleanup_errors = ignore_cleanup_errors

        self.name = mkdtemp(suffix, prefix, dir)

    def __enter__(self) -> String:
        """The function to call when entering the context.

        Returns:
            The temporary directory name.
        """
        return self.name

    def __exit__(self) raises:
        """Called when exiting the context with no error.

        Raises:
            If the operation fails.
        """
        _rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)

    def __exit__(self, err: Error) -> Bool:
        """Called when exiting the context with an error.

        Args:
            err: The error raised inside the context.

        Returns:
            True if the temporary directory was removed successfully.
        """
        try:
            self.__exit__()
            return True
        except:
            return False


struct NamedTemporaryFile(Movable):
    """Temporary file with automatic cleanup.

    Creates a file that's deleted when closed (by default):
    ```mojo
    from std.tempfile import NamedTemporaryFile

    with NamedTemporaryFile(mode="rw") as f:
        f.write("Hello world!")
        f.seek(0)
        print(f.read())  # "Hello world!"
        print(f.name)    # Access path while open
    # File deleted automatically
    ```

    Set `delete=False` to keep the file after closing.
    Use `mode="r"` for reading, `mode="w"` for writing, `mode="rw"` for both.
    """

    var _file_handle: FileHandle
    """The underlying file handle."""
    var _delete: Bool
    """Whether the file is deleted on close."""
    var name: String
    """Name of the file."""

    def __init__(
        out self,
        mode: String = "w",
        name: Optional[String] = None,
        suffix: String = "",
        prefix: String = "tmp",
        dir: Optional[String] = None,
        delete: Bool = True,
    ) raises:
        """Create a named temporary file.

        This is a wrapper around a `FileHandle`,
        `os.remove()` is called in the `close()` method if `delete` is True.

        Can be used as a context manager. When used as a context manager, the
        `close()` is called when the context manager exits.

        Args:
            mode: The mode to open the file in (the mode can be "r" or "w").
            name: The name of the temp file. If it is unspecified, then a random name will be provided.
            suffix: Suffix to use for the file name if name is not provided.
            prefix: Prefix to use for the file name if name is not provided.
            dir: Directory in which the file will be created.
            delete: Whether the file is deleted on close.

        Raises:
            If the operation fails.
        """

        var final_dir = dir.value() if dir else _get_default_tempdir()

        self._delete = delete
        self.name = ""

        if name:
            self.name = name.value()
        else:
            for _ in range(TMP_MAX):
                var potential_name = (
                    final_dir
                    + std.os.sep
                    + prefix
                    + _get_random_name()
                    + suffix
                )
                if not std.os.path.exists(potential_name):
                    self.name = potential_name
                    break
        try:
            # TODO for now this name could be relative,
            # python implementation expands the path,
            # but several functions are not yet implemented in mojo
            # i.e. abspath, normpath
            self._file_handle = FileHandle(self.name, mode=mode)
            return
        except:
            raise Error("Failed to create temporary file")

    def __deinit__(deinit self):
        """Closes the file handle."""
        try:
            self.close()
        except:
            pass

    def close(mut self) raises:
        """Closes the file handle.

        Raises:
            If the operation fails.

        Example:

        ```mojo
        from std.tempfile import NamedTemporaryFile
        import std.os

        var temp_file = NamedTemporaryFile() # delete=True by default
        temp_file.write("Temporary data")
        temp_file.close() # File is deleted if delete=True
        print(std.os.path.exists(temp_file.name)) # False

        temp_file = NamedTemporaryFile(delete=False)
        temp_file.write("Temporary data")
        temp_file.close() # File is not deleted
        print(std.os.path.exists(temp_file.name)) # True
        std.os.remove(temp_file.name) # Clean up manually
        ```
        """
        self._file_handle.close()
        if self._delete:
            std.os.remove(self.name)

    def read(self, size: Int = -1) raises -> String:
        """Reads the data from the file.

        Args:
            size: Requested number of bytes to read.

        Returns:
            The contents of the file.

        Raises:
            If the operation fails.

        Example:

        ```mojo
        from std.tempfile import NamedTemporaryFile
        from std.pathlib import Path

        var p: Path
        with NamedTemporaryFile(mode="rw") as f:
            p = f.name.copy()
            f.write("Hello world!")
            _ = f.seek(0)
            print(
                f.read() == "Hello world!"
            )
        print(p.exists()) # False
        ```
        """
        return self._file_handle.read(size)

    def read_bytes(self, size: Int = -1) raises -> List[UInt8]:
        """Read from file buffer until we have `size` characters or we hit EOF.
        If `size` is negative or omitted, read until EOF.

        Args:
            size: Requested number of bytes to read.

        Returns:
            The contents of the file.

        Raises:
            If the operation fails.

        Example:

        ```mojo
        from std.tempfile import NamedTemporaryFile
        from std.pathlib import Path

        var p: Path
        var bytes = [Byte(0x48), 0x65, 0x6C, 0x6C, 0x6F] # "Hello"

        with NamedTemporaryFile(mode="rw") as f:
            p = f.name.copy()
            f.write_bytes(bytes[:])
            _ = f.seek(0)

            var b = f.read_bytes()
            print(b == bytes) # True

            var s = String(from_utf8_lossy=b)
            print(s) # "Hello"

        print(p.exists()) # False
        ```
        """
        return self._file_handle.read_bytes(size)

    def seek(
        self, offset: Int, whence: UInt8 = std.os.SEEK_SET
    ) raises -> UInt64:
        """Seeks to the given offset in the file.

        Args:
            offset: The byte offset to seek to, relative to `whence`. May be
                negative when `whence` is `os.SEEK_CUR` or `os.SEEK_END`.
            whence: The reference point for the offset:
                os.SEEK_SET = 0: start of file (Default).
                os.SEEK_CUR = 1: current position.
                os.SEEK_END = 2: end of file.

        Raises:
            An error if this file handle is invalid, or if file seek returned a
            failure.

        Returns:
            The resulting byte offset from the start of the file.

        Example:

        ```mojo
        from std.tempfile import NamedTemporaryFile
        from std.pathlib import Path

        var p: Path
        var bytes = [Byte(0x48), 0x65, 0x6C, 0x6C, 0x6F] # "Hello"

        with NamedTemporaryFile(mode="rw") as f:
            p = f.name.copy()
            f.write_bytes(bytes[:])
            _ = f.seek(0)

            var b = f.read_bytes()
            print(b == bytes) # True
            print(String(from_utf8_lossy=b)) # "Hello"
        ```
        """
        return self._file_handle.seek(offset, whence)

    def write[*Ts: Writable](mut self, *args: *Ts):
        """Write a sequence of Writable arguments to the provided Writer.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this Writer.

        Example:
        ```mojo
        from std.tempfile import NamedTemporaryFile
        from std.pathlib import Path

        var p: Path
        with NamedTemporaryFile(mode="rw") as f:
            p = f.name
            f.write("Hello world!")
            _ = f.seek(0)
            print(
                f.read() == "Hello world!"
            )
        ```
        """
        var file = FileDescriptor(self._file_handle._get_raw_fd())
        var buffer = _FlushingWriteBuffer(file)

        comptime for i in range(args.__len__()):
            args[i].write_to(buffer)

        buffer.flush()

    @always_inline
    def write_bytes(mut self, bytes: Span[Byte, _]):
        """
        Write a span of bytes to the file.

        Args:
            bytes: The byte span to write to this file.

        Example:

        ```mojo
        from std.tempfile import NamedTemporaryFile

        var bytes = [Byte(0x48), 0x65, 0x6C, 0x6C, 0x6F]  # "Hello" in ASCII

        with NamedTemporaryFile(mode="rw") as f:
            f.write_bytes(bytes[:])
            _ = f.seek(0)

            var b = f.read_bytes()
            print(b == bytes) # True

            var s = String(from_utf8_lossy=b)
            print(s) # "Hello"
        ```
        """
        self._file_handle.write_bytes(bytes)

    def __enter__(var self) -> Self:
        """The function to call when entering the context.

        Returns:
            The file handle."""
        return self^
