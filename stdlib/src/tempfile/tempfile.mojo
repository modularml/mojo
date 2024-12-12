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
"""Implements tempfile methods.

You can import a method from the `tempfile` package. For example:

```mojo
from tempfile import gettempdir
```
"""

import os
import sys
from collections import List, Optional
from pathlib import Path

from memory import Span
from utils import write_buffered

alias TMP_MAX = 10_000


fn _get_random_name(size: Int = 8) -> String:
    alias characters = String("abcdefghijklmnopqrstuvwxyz0123456789_")
    var name_list = List[UInt8](capacity=size + 1)
    for _ in range(size):
        var rand_index = int(
            random.random_ui64(0, characters.byte_length() - 1)
        )
        name_list.append(ord(characters[rand_index]))
    name_list.append(0)
    return String(name_list^)


fn _candidate_tempdir_list() -> List[String]:
    """Generate a list of candidate temporary directories which
    _get_default_tempdir will try."""

    constrained[not sys.os_is_windows(), "windows not supported yet"]()

    var dirlist = List[String]()
    var possible_env_vars = List("TMPDIR", "TEMP", "TMP")
    var dirname: String

    # First, try the environment.
    for env_var in possible_env_vars:
        if dirname := os.getenv(env_var[]):
            dirlist.append(dirname^)

    # Failing that, try OS-specific locations.
    dirlist.extend(List(String("/tmp"), String("/var/tmp"), String("/usr/tmp")))

    # As a last resort, the current directory if possible,
    # os.path.getcwd() could raise
    try:
        dirlist.append(str(Path()))
    except:
        pass

    return dirlist


fn _get_default_tempdir() raises -> String:
    """Calculate the default directory to use for temporary files.

    We determine whether or not a candidate temp dir is usable by
    trying to create and write to a file in that directory.  If this
    is successful, the test file is deleted. To prevent denial of
    service, the name of the test file must be randomized."""

    var dirlist = _candidate_tempdir_list()

    for dir_name in dirlist:
        if not os.path.isdir(dir_name[]):
            continue
        if _try_to_create_file(dir_name[]):
            return dir_name[]

    raise Error("No usable temporary directory found")


fn _try_to_create_file(dir: String) -> Bool:
    for _ in range(TMP_MAX):
        var name = _get_random_name()
        # TODO use os.join when it exists
        var filename = Path(dir) / name

        # prevent overwriting existing file
        if os.path.exists(filename):
            continue

        # verify that we have writing access in the target directory
        try:
            with FileHandle(str(filename), "w"):
                pass
            os.remove(filename)
            return True
        except:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass
            return False

    return False


fn gettempdir() -> Optional[String]:
    """Return the default directory to use for temporary files.

    Returns:
        The name of the default temporary directory.
    """
    # TODO In python _get_default_tempdir is called exactly once so that the default
    # tmp dir is the same throughout the program execution/
    # Since there is no global scope in mojo yet, this is not possible for now.
    try:
        return _get_default_tempdir()
    except:
        return None


fn mkdtemp(
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
    """
    var final_dir = Path(dir.value()) if dir else Path(_get_default_tempdir())

    for _ in range(TMP_MAX):
        var dir_name = final_dir / (prefix + _get_random_name() + suffix)
        if os.path.exists(dir_name):
            continue
        try:
            os.mkdir(dir_name, mode=0o700)
            # TODO for now this name could be relative,
            # python implementation expands the path,
            # but several functions are not yet implemented in mojo
            # i.e. abspath, normpath
            return str(dir_name)
        except:
            continue
    raise Error("Failed to create temporary file")


# TODO use shutil.rmtree (or equivalent) when it exists
fn _rmtree(path: String, ignore_errors: Bool = False) raises:
    """Removes the specified directory and all its contents.

    If the path is a symbolic link, an error is raised. If ignore_errors is
    True, errors resulting from failed removals will be ignored. Absolute and
    relative paths are allowed, relative paths are resolved from cwd.

    Args:
        path: The path to the directory.
        ignore_errors: Whether to ignore errors.
    """
    if os.path.islink(path):
        raise Error("`path`can not be a symbolic link: " + path)

    for file_or_dir in os.listdir(path):
        var curr_path = os.path.join(path, file_or_dir[])
        if os.path.isfile(curr_path):
            try:
                os.remove(curr_path)
            except e:
                if not ignore_errors:
                    raise e
            continue
        if os.path.isdir(curr_path):
            try:
                _rmtree(curr_path, ignore_errors)
            except e:
                if ignore_errors:
                    continue
                raise e
    try:
        os.rmdir(path)
    except e:
        if not ignore_errors:
            raise e


struct TemporaryDirectory:
    """A temporary directory."""

    var name: String
    """The name of the temporary directory."""
    var _ignore_cleanup_errors: Bool
    """Whether to ignore cleanup errors."""

    fn __init__(
        mut self,
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
        """
        self._ignore_cleanup_errors = ignore_cleanup_errors

        self.name = mkdtemp(suffix, prefix, dir)

    fn __enter__(self) -> String:
        """The function to call when entering the context.

        Returns:
            The temporary directory name.
        """
        return self.name

    fn __exit__(self) raises:
        """Called when exiting the context with no error."""
        _rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)

    fn __exit__(self, err: Error) -> Bool:
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


struct NamedTemporaryFile:
    """A handle to a temporary file.

    Example:
    ```mojo
    from tempfile import NamedTemporaryFile
    from pathlib import Path
    def main():
        var p: Path
        with NamedTemporaryFile(mode="rw") as f:
            p = f.name
            f.write("Hello world!")
            f.seek(0)
            print(
                f.read() == "Hello world!"
            )
        print(str(p), p.exists()) #Removed by default
    ```
    Note: `NamedTemporaryFile.__init__` document the arguments.
    """

    var _file_handle: FileHandle
    """The underlying file handle."""
    var _delete: Bool
    """Whether the file is deleted on close."""
    var name: String
    """Name of the file."""

    fn __init__(
        mut self,
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
        """

        var final_dir = dir.value() if dir else _get_default_tempdir()

        self._delete = delete
        self.name = ""

        if name:
            self.name = name.value()
        else:
            for _ in range(TMP_MAX):
                var potential_name = final_dir + os.sep + prefix + _get_random_name() + suffix
                if not os.path.exists(potential_name):
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

    fn __del__(owned self):
        """Closes the file handle."""
        try:
            self.close()
        except:
            pass

    fn close(mut self) raises:
        """Closes the file handle."""
        self._file_handle.close()
        if self._delete:
            os.remove(self.name)

    fn __moveinit__(out self, owned existing: Self):
        """Moves constructor for the file handle.

        Args:
            existing: The existing file handle.
        """
        self._file_handle = existing._file_handle^
        self._delete = existing._delete
        self.name = existing.name^

    fn read(self, size: Int64 = -1) raises -> String:
        """Reads the data from the file.

        Args:
            size: Requested number of bytes to read.

        Returns:
            The contents of the file.
        """
        return self._file_handle.read(size)

    fn read_bytes(self, size: Int64 = -1) raises -> List[UInt8]:
        """Read from file buffer until we have `size` characters or we hit EOF.
        If `size` is negative or omitted, read until EOF.

        Args:
            size: Requested number of bytes to read.

        Returns:
            The contents of the file.
        """
        return self._file_handle.read_bytes(size)

    fn seek(self, offset: UInt64, whence: UInt8 = os.SEEK_SET) raises -> UInt64:
        """Seeks to the given offset in the file.

        Args:
            offset: The byte offset to seek to from the start of the file.
            whence: The reference point for the offset:
                os.SEEK_SET = 0: start of file (Default).
                os.SEEK_CUR = 1: current position.
                os.SEEK_END = 2: end of file.

        Raises:
            An error if this file handle is invalid, or if file seek returned a
            failure.

        Returns:
            The resulting byte offset from the start of the file.
        """
        return self._file_handle.seek(offset, whence)

    fn write[*Ts: Writable](mut self, *args: *Ts):
        """Write a sequence of Writable arguments to the provided Writer.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this Writer.
        """
        var file = FileDescriptor(self._file_handle._get_raw_fd())
        write_buffered[buffer_size=4096](file, args)

    @always_inline
    fn write_bytes(mut self, bytes: Span[Byte, _]):
        """
        Write a span of bytes to the file.

        Args:
            bytes: The byte span to write to this file.
        """
        self._file_handle.write_bytes(bytes)

    fn __enter__(owned self) -> Self:
        """The function to call when entering the context.

        Returns:
            The file handle."""
        return self^
