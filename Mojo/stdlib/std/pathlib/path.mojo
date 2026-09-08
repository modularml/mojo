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
"""Path manipulation module.

This module defines a platform-independent API for working with filesystem
paths. `Path`, its core type, represents a filesystem path. It exposes
operations such as path composition, existence checks, file I/O, and
access to file attributes.

To use these features import the `Path` type from this module.

Example:

```mojo
from std.pathlib import Path
var p = Path("a") / "b" / "c.txt"
print(p)  # a/b/c.txt
```
"""

import std.os
import std.format._utils as fmt
from std.hashlib.hasher import Hasher
from std.os import PathLike, listdir, stat_result
from std.ffi import c_char, external_call
from std.sys import CompilationTarget

from std.reflection import call_location

comptime DIR_SEPARATOR = "/"
"""The directory separator character for path operations."""


def cwd() raises -> Path:
    """Gets the current directory.

    Returns:
      The current directory.

    Raises:
        If the operation fails.

    Example:

    ```mojo
    from std.pathlib import cwd

    var string_path = cwd()
    print(string_path)
    ```
    """
    comptime MAX_CWD_BUFFER_SIZE = 1024
    var buf = Array[c_char, MAX_CWD_BUFFER_SIZE](uninitialized=True)

    var ptr = buf.unsafe_ptr()
    var res = external_call["getcwd", OptionalPointer[c_char, origin_of(buf)]](
        ptr, Int(MAX_CWD_BUFFER_SIZE)
    )

    # If we get a nullptr, then we raise an error.
    if not res:
        raise Error("unable to query the current directory")

    return String(unsafe_from_utf8_ptr=ptr)


@always_inline
def _dir_of_current_file() raises -> Path:
    """Gets the directory the file is at.

    Returns:
      The directory the file calling is at.
    """
    return _dir_of_current_file_impl(call_location().file_name())


@no_inline
def _dir_of_current_file_impl(file_name: StaticString) raises -> Path:
    var i = String(file_name).rfind(DIR_SEPARATOR)
    return Path(file_name[byte=0:i])


struct Path(
    Boolable,
    Comparable,
    Hashable,
    ImplicitlyCopyable,
    KeyElement,
    PathLike,
    Writable,
):
    """The Path object."""

    var path: String
    """The underlying path string representation."""

    def __init__(out self) raises:
        """Initializes a path with the current directory.

        Raises:
            If the operation fails.
        """
        self = cwd()

    # Note: Not @implicit so that allocation is not implicit.
    def __init__(out self, path: StringSlice):
        """Initializes a path with the provided path.

        Args:
          path: The file system path.
        """
        self.path = String(path)

    @implicit
    def __init__(out self, var path: String):
        """Initializes a path with the provided path.

        Args:
          path: The file system path.
        """
        self.path = path^

    @implicit
    def __init__(out self, path: StringLiteral):
        """Initializes a path with the provided path.

        Args:
          path: The file system path.
        """
        self.path = path

    def __truediv__(self, suffix: Self) -> Self:
        """Joins two paths using the system-defined path separator.

        Args:
          suffix: The suffix to append to the path.

        Returns:
          A new path with the suffix appended to the current path.
        """
        return self.__truediv__(StringSlice(suffix.path))

    def __truediv__(self, suffix: StringSlice) -> Self:
        """Joins two paths using the system-defined path separator.

        Args:
          suffix: The suffix to append to the path.

        Returns:
          A new path with the suffix appended to the current path.
        """
        var res = self
        res /= suffix
        return res

    def __itruediv__(mut self, suffix: StringSlice):
        """Joins two paths using the system-defined path separator.

        Args:
          suffix: The suffix to append to the path.
        """
        if self.path.endswith(DIR_SEPARATOR):
            self.path += suffix
        else:
            self.path += DIR_SEPARATOR
            self.path += suffix

    @always_inline
    def __bool__(self) -> Bool:
        """Checks if the path is not empty.

        Returns:
            True if the path length is greater than zero, and False otherwise.
        """
        return self.path.byte_length() > 0

    def write_to(self, mut writer: Some[Writer]):
        """
        Formats this path to the provided Writer.

        Args:
            writer: The object to write to.
        """

        writer.write(self.path)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the repr of this `Path` to a writer.

        Writes the path in the format `Path('...')`.

        Args:
            writer: The object to write to.
        """
        fmt.FormatStruct(writer, "Path").fields(fmt.Repr(self.path))

    @always_inline
    def __fspath__(self) -> String:
        """Returns a string representation of the path.

        Returns:
          A string representation of the path.
        """
        return self.path

    def __eq__(self, other: Self) -> Bool:
        """Returns True if the two paths are equal.

        Args:
          other: The other path to compare against.

        Returns:
          True if the paths are equal and False otherwise.
        """
        return String(self) == String(other)

    def __eq__(self, other: StringSlice) -> Bool:
        """Returns True if the two paths are equal.

        Args:
          other: The other path to compare against.

        Returns:
          True if the String and Path are equal, and False otherwise.
        """
        return StringSlice(self.path) == other

    @always_inline
    def __lt__(self, other: Self) -> Bool:
        """Returns True if this path is less than the other path.

        Comparison uses lexicographic ordering of the underlying path strings.

        Args:
          other: The other path to compare against.

        Returns:
          True if this path is less than the other path, and False otherwise.
        """
        return self.path < other.path

    def stat(self) raises -> stat_result:
        """Returns the stat information on the path.

        Returns:
          A stat_result object containing information about the path.

        Raises:
            If the operation fails.

        Example:

        ```mojo
        from std.pathlib import Path
        var p = Path()       # Path to cwd
        print(p.stat())      # os.stat_result(...)
        ```
        """
        return std.os.stat(self)

    def lstat(self) raises -> stat_result:
        """Returns the lstat information on the path. This is similar to stat,
        but if the file is a symlink then it gives you information about the
        symlink rather than the target.

        Returns:
          A stat_result object containing information about the path.

        Raises:
            If the operation fails.
        """
        return std.os.lstat(self)

    @always_inline
    def exists(self) -> Bool:
        """Returns True if the path exists and False otherwise.

        Returns:
          True if the path exists on disk and False otherwise.

        Example:

        ```mojo
        from std.pathlib import Path

        var p = Path("./path/to/nowhere/does-not-exist")
        print("Exists" if p.exists() else "Does not exist") # Does not exist
        ```
        """
        return std.os.path.exists(self)

    def expanduser(self) raises -> Path:
        """Expands a prefixed `~` with `$HOME` on posix
        If environment variables are not set or the `path` is not
        prefixed with `~`, returns the `path` unmodified.

        Returns:
            The expanded path.

        Raises:
            If the operation fails.

        Example:

        ```mojo
        from std.pathlib import Path
        from std.testing import assert_true

        var p = Path("~")
        assert_true(p.expanduser() == Path.home())
        ```
        """
        return std.os.path.expanduser(self)

    @staticmethod
    def home() raises -> Path:
        """Returns `$HOME` on posix.
        If environment variables are not set it returns `~`.

        Returns:
            Path to user home directory.

        Raises:
            If the operation fails.

        Example:

        ```mojo
        from std.pathlib import Path
        from std.testing import assert_true

        var p = Path("~")
        assert_true(p.expanduser() == Path.home())
        ```
        """
        return std.os.path.expanduser("~")

    def is_dir(self) -> Bool:
        """Returns True if the path is a directory and False otherwise.

        Returns:
          Return True if the path points to a directory (or a link pointing to
          a directory).

        Example:

        ```mojo
        from std.pathlib import Path
        from std.testing import assert_true

        var p = Path.home()
        assert_true(p.is_dir())
        ```
        """
        return std.os.path.isdir(self)

    def is_file(self) -> Bool:
        """Returns True if the path is a file and False otherwise.

        Returns:
          Return True if the path points to a file (or a link pointing to
          a file).

        Example:

        ```mojo
        from std.pathlib import Path
        from std.testing import assert_false

        var p = Path.home()
        assert_false(p.is_file())
        ```
        """
        return std.os.path.isfile(self)

    def read_text(self) raises -> String:
        """Returns content of the file.

        Returns:
          Contents of file as string.

        Raises:
            If the operation fails.

        Example:

        ```mojo
        from std.pathlib import Path

        var p = Path("testfile.txt")
        p.write_text("Hello Mojo")
        if p.exists():
            var contents = p.read_text()
            print(contents) # Hello Mojo
        ```
        """
        with open(self, "r") as f:
            return f.read()

    def read_bytes(self) raises -> List[Byte]:
        """Returns content of the file as bytes.

        Returns:
          Contents of file as list of bytes.

        Raises:
            If the operation fails.

        Example:

        ```mojo
        from std.pathlib import Path
        from std.testing import assert_true

        var p = Path("testfile.txt")
        p.write_text("test")
        if p.exists():
            var contents = p.read_bytes()
            assert_true(contents[0] == 116)
        ```
        """
        with open(self, "r") as f:
            return f.read_bytes()

    def write_text[T: Writable](self, value: T) raises:
        """Writes the value to the file as text.

        Parameters:
            T: The type of an object conforming to the `Writable` trait.

        Args:
            value: The value to write.

        Raises:
            If the operation fails.

        Example:

        ```mojo
        from std.pathlib import Path

        var p = Path("testfile")
        p.write_text("Hello")
        if p.exists():
            var contents = p.read_text()
            print(contents) # Hello
        ```
        """
        with open(self, "w") as f:
            f.write(value)

    def write_bytes(self, bytes: Span[Byte, _]) raises:
        """Writes bytes to the file.

        Args:
            bytes: The bytes to write to this file.

        Raises:
            If the operation fails.

        Example:

        ```mojo
        from std.pathlib import Path

        var p = Path("testfile")
        var s = "Hello"
        p.write_bytes(s.as_bytes())
        if p.exists():
            var contents = p.read_text()
            print(contents) # Hello
        ```
        """
        with open(self, "w") as f:
            f.write_bytes(bytes)

    def suffix(self) -> String:
        """The path's extension, if any.
        This includes the leading period. For example: '.txt'.
        If no extension is found, returns the empty string.

        Returns:
            The path's extension.

        Example:

        ```mojo
        from std.pathlib import Path
        from std.testing import assert_true

        var p = Path("testfile.txt")
        print(p.suffix())
        assert_true(p.suffix() == ".txt")

        p = Path(".hiddenfile")
        assert_true(p.suffix() == "") # No suffix
        ```
        """
        # +2 to skip both `DIR_SEPARATOR` and the first ".".
        # For example /a/.foo's suffix is "" but /a/b.foo's suffix is .foo.
        var start = self.path.rfind(DIR_SEPARATOR) + 2
        var i = self.path.rfind(".", start)
        if 0 < i < (self.path.byte_length() - 1):
            return String(self.path[byte=i:])

        return ""

    # TODO(MOCO-1532):
    #   Use StringSlice here once param inference bug for empty variadic
    #   list of parameterized types is fixed.
    def joinpath(self, *pathsegments: String) -> Path:
        """Joins the Path using the pathsegments.

        Args:
            pathsegments: The path segments.

        Returns:
            The path concatenation with the pathsegments using the
            directory separator.

        Example:

        ```mojo
        from std.pathlib import Path
        from std.tempfile import gettempdir
        from std.testing import assert_true

        # gettmpdir() has no guarantee of trailing /
        # Use joinpath to ensure path construction
        var p = Path("/tmp")
        p = p.joinpath("testdir")  # No trailing /
        p = p.joinpath("testfile.txt")
        assert_true(p == Path("/tmp/testdir/testfile.txt"))

        p = Path("/tmp/")
        p = p.joinpath("testdir/")  # Trailing /
        p = p.joinpath("testfile.txt")
        assert_true(p == Path("/tmp/testdir/testfile.txt"))
        ```
        """
        if len(pathsegments) == 0:
            return self

        var result = self

        for i in range(len(pathsegments)):
            result /= pathsegments[i]

        return result

    def listdir(self) raises -> List[Path]:
        """Gets the list of entries contained in the path provided.

        Returns:
            The list of entries in the path provided.

        Raises:
            If the operation fails.

        Example:

        ```mojo
        from std.pathlib import Path, cwd

        for item in cwd().listdir():
            print(item) # each item name in working directory
        ```
        """

        var ls = listdir(self)
        return List(length=len(ls), fill_with=lambda (i: Int) -> Path: ls[i])

    def name(self) -> String:
        """Returns the name of the path.

        Returns:
            The name of the path.

        Example:

        ```mojo
        from std.pathlib import Path

        Path("a/path/foo.txt").name()  # returns "foo.txt"
        ```
        """
        return std.os.path.basename(self)

    def parts(
        self,
    ) -> List[StringSlice[origin_of(self.path)._get_owned_interior["bytes"]]]:
        """Returns the parts of the path separated by `DIR_SEPARATOR`.

        Returns:
            The parts of the path separated by `DIR_SEPARATOR`.

        Example:

        ```mojo
        from std.pathlib import Path
        from std.testing import assert_true

        for p, q in zip(Path("a/path/foo.txt").parts(), ["a", "path", "foo.txt"]):
            assert_true(p == q)
        ```
        """
        return self.path.split(DIR_SEPARATOR)
