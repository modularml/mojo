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
"""Implements `Path` and related functions.
"""

import os
from os import PathLike, listdir, stat_result
from sys import os_is_windows
from builtin._location import __call_location

from memory import stack_allocation

from utils import StringRef

from sys.ffi import C_char

alias DIR_SEPARATOR = "\\" if os_is_windows() else "/"


fn cwd() raises -> Path:
    """Gets the current directory.

    Returns:
      The current directory.
    """
    alias MAX_CWD_BUFFER_SIZE = 1024
    var buf0 = stack_allocation[MAX_CWD_BUFFER_SIZE, C_char.type]()

    var buf = UnsafePointer[C_char]._from_dtype_ptr(buf0)

    var res = external_call["getcwd", UnsafePointer[C_char]](
        buf, MAX_CWD_BUFFER_SIZE
    )

    # If we get a nullptr, then we raise an error.
    if res == UnsafePointer[C_char]():
        raise Error("unable to query the current directory")

    return String(StringRef(buf))


@always_inline
fn _dir_of_current_file() raises -> Path:
    """Gets the directory the file is at.

    Returns:
      The directory the file calling is at.
    """
    var file_name = __call_location().file_name
    var i = str(file_name).rfind(DIR_SEPARATOR)
    return Path(str(file_name)[0:i])


struct Path(Stringable, CollectionElement, PathLike, KeyElement):
    """The Path object."""

    var path: String
    """The underlying path string representation."""

    fn __init__(inout self) raises:
        """Initializes a path with the current directory."""
        self = cwd()

    fn __init__(inout self, path: String):
        """Initializes a path with the provided path.

        Args:
          path: The file system path.
        """
        self.path = path

    fn __moveinit__(inout self, owned existing: Self):
        """Move data of an existing Path into a new one.

        Args:
            existing: The existing Path.
        """
        self.path = existing.path^

    @always_inline
    fn __copyinit__(inout self, existing: Self):
        """Copy constructor for the path struct.

        Args:
          existing: The existing struct to copy from.
        """
        self.path = existing.path

    fn __truediv__(self, suffix: Self) -> Self:
        """Joins two paths using the system-defined path separator.

        Args:
          suffix: The suffix to append to the path.

        Returns:
          A new path with the suffix appended to the current path.
        """
        return self.__truediv__(suffix.path)

    fn __truediv__(self, suffix: String) -> Self:
        """Joins two paths using the system-defined path separator.

        Args:
          suffix: The suffix to append to the path.

        Returns:
          A new path with the suffix appended to the current path.
        """
        var res = self
        res /= suffix
        return res

    fn __itruediv__(inout self, suffix: String):
        """Joins two paths using the system-defined path separator.

        Args:
          suffix: The suffix to append to the path.
        """
        if self.path.endswith(DIR_SEPARATOR):
            self.path += suffix
        else:
            self.path += DIR_SEPARATOR + suffix

    @always_inline
    fn __str__(self) -> String:
        """Returns a string representation of the path.

        Returns:
          A string representation of the path.
        """
        return self.path

    @always_inline
    fn __fspath__(self) -> String:
        """Returns a string representation of the path.

        Returns:
          A string representation of the path.
        """
        return str(self)

    fn __repr__(self) -> String:
        """Returns a printable representation of the path.

        Returns:
          A printable representation of the path.
        """
        return str(self)

    fn __eq__(self, other: Self) -> Bool:
        """Returns True if the two paths are equal.

        Args:
          other: The other path to compare against.

        Returns:
          True if the paths are equal and False otherwise.
        """
        return str(self) == str(other)

    fn __ne__(self, other: Self) -> Bool:
        """Returns True if the two paths are not equal.

        Args:
          other: The other path to compare against.

        Returns:
          True if the paths are not equal and False otherwise.
        """
        return not self == other

    fn __hash__(self) -> Int:
        """Hash the underlying path string using builtin hash.

        Returns:
            An integer value containing the hash of the path string.
        """

        return hash(self.path)

    fn stat(self) raises -> stat_result:
        """Returns the stat information on the path.

        Returns:
          A stat_result object containing information about the path.
        """
        return os.stat(self)

    fn lstat(self) raises -> stat_result:
        """Returns the lstat information on the path. This is similar to stat,
        but if the file is a symlink then it gives you information about the
        symlink rather than the target.

        Returns:
          A stat_result object containing information about the path.
        """
        return os.lstat(self)

    @always_inline
    fn exists(self) -> Bool:
        """Returns True if the path exists and False otherwise.

        Returns:
          True if the path exists on disk and False otherwise.
        """
        return os.path.exists(self)

    fn is_dir(self) -> Bool:
        """Returns True if the path is a directory and False otherwise.

        Returns:
          Return True if the path points to a directory (or a link pointing to
          a directory).
        """
        return os.path.isdir(self)

    fn is_file(self) -> Bool:
        """Returns True if the path is a file and False otherwise.

        Returns:
          Return True if the path points to a file (or a link pointing to
          a file).
        """
        return os.path.isfile(self)

    fn read_text(self) raises -> String:
        """Returns content of the file.

        Returns:
          Contents of file as string.
        """
        with open(self, "r") as f:
            return f.read()

    fn read_bytes(self) raises -> List[UInt8]:
        """Returns content of the file as bytes.

        Returns:
          Contents of file as list of bytes.
        """
        with open(self, "r") as f:
            return f.read_bytes()

    fn write_text[stringable: Stringable](self, value: stringable) raises:
        """Writes the value to the file as text.

        Parameters:
          stringable: The Stringable type.

        Args:
          value: The value to write.
        """
        with open(self, "w") as f:
            f.write(str(value))

    @always_inline
    fn suffix(self) -> String:
        """The path's extension, if any.
        This includes the leading period. For example: '.txt'.
        If no extension is found, returns the empty string.

        Returns:
            The path's extension.
        """
        # +2 to skip both `DIR_SEPARATOR` and the first ".".
        # For example /a/.foo's suffix is "" but /a/b.foo's suffix is .foo.
        var start = self.path.rfind(DIR_SEPARATOR) + 2
        var i = self.path.rfind(".", start)
        if 0 < i < (len(self.path) - 1):
            return self.path[i:]

        return ""

    fn joinpath(self, *pathsegments: String) -> Path:
        """Joins the Path using the pathsegments.

        Args:
            pathsegments: The path segments.

        Returns:
            The path concatenation with the pathsegments using the
            directory separator.
        """
        if len(pathsegments) == 0:
            return self

        var result = self

        for i in range(len(pathsegments)):
            result /= pathsegments[i]

        return result

    fn listdir(self) raises -> List[Path]:
        """Gets the list of entries contained in the path provided.

        Returns:
            The list of entries in the path provided.
        """

        var ls = listdir(self)
        var res = List[Path](capacity=len(ls))
        for i in range(len(ls)):
            res.append(ls[i])

        return res
