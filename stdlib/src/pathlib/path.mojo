# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements `Path` and related functions.
"""

from sys.info import os_is_windows

from memory import stack_allocation
from collections.vector import CollectionElement
from tensor import Tensor

alias DIR_SEPARATOR = "\\" if os_is_windows() else "/"


fn cwd() raises -> Path:
    """Gets the current directory.

    Returns:
      The current directory.
    """
    alias MAX_CWD_BUFFER_SIZE = 1024
    let buf = stack_allocation[MAX_CWD_BUFFER_SIZE, DType.int8]()

    let res = external_call["getcwd", DTypePointer[DType.int8]](
        buf, MAX_CWD_BUFFER_SIZE
    )

    # If we get a nullptr, then we raise an error.
    if res == DTypePointer[DType.int8]():
        raise Error("unable to query the current directory")

    return String(buf)


struct Path(Stringable, CollectionElement):
    """The Path object."""

    var path: String
    """The underlying path string representation."""

    fn __init__(inout self) raises:
        """Initializes a path with the current directory."""
        self = cwd()

    fn __init__(inout self, path: StringLiteral):
        """Initializes a path with the provided path.

        Args:
          path: The file system path.
        """
        self.path = path

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
        self.path = existing.path ^

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

    fn __truediv__(self, suffix: StringLiteral) -> Self:
        """Joins two paths using the system-defined path separator.

        Args:
          suffix: The suffix to append to the path.

        Returns:
          A new path with the suffix appended to the current path.
        """
        return self.__truediv__(String(suffix))

    fn __truediv__(self, suffix: String) -> Self:
        """Joins two paths using the system-defined path separator.

        Args:
          suffix: The suffix to append to the path.

        Returns:
          A new path with the suffix appended to the current path.
        """
        if self.path.endswith(DIR_SEPARATOR):
            return self.path + suffix
        return self.path + DIR_SEPARATOR + suffix

    fn __str__(self) -> String:
        """Returns a string representation of the path.

        Returns:
          A string represntation of the path.
        """
        return self.path

    fn __repr__(self) -> String:
        """Returns a printable representation of the path.

        Returns:
          A printable represntation of the path.
        """
        return self.__str__()

    fn __eq__(self, other: Self) -> Bool:
        """Returns True if the two paths are equal.

        Args:
          other: The other path to compare against.

        Returns:
          True if the paths are equal and False otherwise.
        """
        return self.__str__() == other.__str__()

    fn __ne__(self, other: Self) -> Bool:
        """Returns True if the two paths are not equal.

        Args:
          other: The other path to compare against.

        Returns:
          True if the paths are not equal and False otherwise.
        """
        return not self == other

    fn exists(self) -> Bool:
        """Returns True if the path exists and False otherwise.

        Returns:
          True if the path exists on disk and False otherwise.
        """
        alias mode = "r"
        let handle = external_call["fopen", Pointer[NoneType]](
            self.path._as_ptr(), mode.data()
        )

        if not handle:
            return False

        _ = external_call["fclose", Int32](handle)
        return True

    fn read_text(self) raises -> String:
        """Returns content of the file.

        Returns:
          Contents of file as string.
        """
        with open(self, "r") as f:
            return f.read()

    fn read_bytes(self) raises -> Tensor[DType.int8]:
        """Returns content of the file as bytes.

        Returns:
          Contents of file as 1D Tensor of bytes.
        """
        with open(self, "r") as f:
            return f.read_bytes()

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
        let start = self.path.rfind(DIR_SEPARATOR) + 2
        let i = self.path.rfind(".", start)
        if 0 < i < (len(self.path) - 1):
            return self.path[i:]

        return ""
