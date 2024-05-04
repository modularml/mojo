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
from tempfile import NamedTemporaryFile
```
"""

from collections import Optional
import os
import sys
import pathlib
from pathlib import Path


fn _get_random_name(size: Int = 8) -> String:
    var characters = String("abcdefghijklmnopqrstuvwxyz0123456789_")
    var name = String("")
    random.seed()
    for _ in range(size):
        var rand_index = int(random.random_ui64(0, len(characters) - 1))
        name += characters[rand_index]
    return name


fn _candidate_tempdir_list() -> List[String]:
    """Generate a list of candidate temporary directories which
    _get_default_tempdir will try."""

    var dirlist = List[String]()
    var possible_env_vars = List("TMPDIR", "TEMP", "TMP")
    var env_var: String
    var dirname: String

    # First, try the environment.
    for env_var in possible_env_vars:
        dirname = os.getenv(env_var[])
        if dirname:
            dirlist.append(dirname)

    # Failing that, try OS-specific locations.
    if sys.os_is_windows():
        # TODO handle windows
        pass
    else:
        dirlist.extend(
            List(String("/tmp"), String("/var/tmp"), String("/usr/tmp"))
        )

    # As a last resort, the current directory.
    try:
        dirlist.append(pathlib.path.cwd())
    except:
        pass

    return dirlist


fn _get_default_tempdir() raises -> String:
    """Calculate the default directory to use for temporary files.

    We determine whether or not a candidate temp dir is usable by
    trying to create and write to a file in that directory.  If this
    is successful, the test file is deleted. To prevent denial of
    service, the name of the test file must be randomized."""
    # TODO In python this function is called exactly one such that the default
    # tmp dir is the same along the program execution,
    # since there is not a global scope in mojo yet this is not possible for now

    var dirlist = _candidate_tempdir_list()
    var dir_name: String

    for dir_name in dirlist:
        if not os.path.isdir(dir_name[]):
            continue
        for _ in range(100):
            var name = _get_random_name()
            var filename = dir_name[] + "/" + name
            if os.path.isfile(filename):
                continue

            try:
                var temp_file = FileHandle(filename, "w")
                temp_file.close()
                os.remove(filename)
                return dir_name[]
            except:
                break
    raise Error("No usable temporary directory found")


struct NamedTemporaryFile:
    """A handle to a temporary file."""

    var _file_handle: FileHandle
    """The underlying file handle."""
    var _delete: Bool
    """Whether the file is deleted on close."""
    var name: String
    """Name of the file."""

    fn __init__(
        inout self,
        mode: String = "w",
        suffix: String = "",
        prefix: String = "tmp",
        dir: Optional[String] = None,
        delete: Bool = True,
    ) raises:
        """Create a named temporary file. Can be used as a context manager.
        This is a wrapper around a `FileHandle`,
        os.remove is called in close method if `delete` is True.

        Args:
            mode: The mode to open the file in (the mode can be "r" or "w").
            suffix: Suffix to use for the file name.
            prefix: Prefix to use for the file name.
            dir: Directory in which the file will be created.
            delete: Whether the file is deleted on close.
        """
        var final_dir: Path
        if not dir:
            final_dir = Path(_get_default_tempdir())
        else:
            final_dir = Path(dir.value()[])

        self._delete = delete

        var MAX_TRIES = 100
        for _ in range(MAX_TRIES):
            var potential_name = final_dir / (
                prefix + _get_random_name() + suffix
            )
            if os.path.exists(potential_name):
                continue
            try:
                self._file_handle = FileHandle(potential_name, mode=mode)
                # TODO for now this name could be relative,
                # python implementation expands the path,
                # but several functions are not yet implemented in mojo
                # i.e. abspath, normpath
                self.name = potential_name
                return
            except:
                continue
        raise Error("Failed to create temporary file")

    @always_inline
    fn __del__(owned self):
        """Closes the file handle."""
        try:
            self.close()
        except:
            pass

    fn close(inout self) raises:
        """Closes the file handle."""
        self._file_handle.close()
        if self._delete:
            os.remove(self.name)

    fn __moveinit__(inout self, owned existing: Self):
        """Moves constructor for the file handle.

        Args:
          existing: The existing file handle.
        """
        self._file_handle = existing._file_handle^
        self._delete = existing._delete
        self.name = existing.name

    @always_inline
    fn read(self, size: Int64 = -1) raises -> String:
        """Reads the data from the file.

        Args:
            size: Requested number of bytes to read.

        Returns:
          The contents of the file.
        """
        return self._file_handle.read(size)

    fn read_bytes(self, size: Int64 = -1) raises -> List[Int8]:
        """Read from file buffer until we have `size` characters or we hit EOF.
        If `size` is negative or omitted, read until EOF.

        Args:
            size: Requested number of bytes to read.

        Returns:
          The contents of the file.
        """
        return self._file_handle.read_bytes(size)

    fn seek(self, offset: UInt64) raises -> UInt64:
        """Seeks to the given offset in the file.

        Args:
            offset: The byte offset to seek to from the start of the file.

        Raises:
            An error if this file handle is invalid, or if file seek returned a
            failure.

        Returns:
            The resulting byte offset from the start of the file.
        """
        return self._file_handle.seek(offset)

    fn write(self, data: String) raises:
        """Write the data to the file.

        Args:
          data: The data to write to the file.
        """
        self._file_handle.write(data)

    fn __enter__(owned self) -> Self:
        """The function to call when entering the context."""
        return self^


struct TemporaryDirectory:
    """A temporary directory."""

    var name: String
    """The name of the temporary directory."""
    var _ignore_cleanup_errors: Bool
    """Whether to ignore cleanup errors."""

    fn __init__(
        inout self,
        suffix: String = "",
        prefix: String = "tmp",
        dir: Optional[String] = None,
        ignore_cleanup_errors: Bool = False,
    ) raises:
        """Create a temporary directory. Can be used as a context manager.

        Args:
            suffix: Suffix to use for the directory name.
            prefix: Prefix to use for the directory name.
            dir: Directory in which the directory will be created.
            ignore_cleanup_errors: Whether to ignore cleanup errors.
        """
        self._ignore_cleanup_errors = ignore_cleanup_errors
        var final_dir: Path
        if not dir:
            final_dir = Path(_get_default_tempdir())
        else:
            final_dir = Path(dir.value()[])

        var MAX_TRIES = 100
        for _ in range(MAX_TRIES):
            var potential_name = final_dir / (
                prefix + _get_random_name() + suffix
            )
            if os.path.exists(potential_name):
                continue
            try:
                os.mkdir(potential_name, mode=0o700)
                # TODO for now this name could be relative,
                # python implementation expands the path,
                # but several functions are not yet implemented in mojo
                # i.e. abspath, normpath
                self.name = potential_name
                return
            except:
                continue
        raise Error("Failed to create temporary file")

    fn __enter__(self) -> String:
        return self.name

    fn __exit__(self) raises:
        shutil.rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)

    fn __exit__(self, err: Error) -> Bool:
        try:
            self.__exit__()
            return True
        except:
            return False
