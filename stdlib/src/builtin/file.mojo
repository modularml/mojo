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
"""Implements the file based methods.

These are Mojo built-ins, so you don't need to import them.

For example, here's how to read a file:

```mojo
var  f = open("my_file.txt", "r")
print(f.read())
f.close()
```

Or use a `with` statement to close the file automatically:

```mojo
with open("my_file.txt", "r") as f:
  print(f.read())
```

"""

from os import PathLike
from sys import external_call

from memory.reference import AddressSpace
from memory.unsafe import DTypePointer


@register_passable
struct _OwnedStringRef(Boolable):
    var data: DTypePointer[DType.int8]
    var length: Int

    fn __init__() -> _OwnedStringRef:
        return Self {data: DTypePointer[DType.int8](), length: 0}

    fn __del__(owned self):
        if self.data:
            self.data.free()

    fn consume_as_error(owned self) -> Error:
        var data = self.data
        # Don't free self.data in our dtor.
        self.data = DTypePointer[DType.int8]()
        var length = self.length
        return Error {data: data, loaded_length: -length}

    fn __bool__(self) -> Bool:
        return self.length != 0


struct FileHandle:
    """File handle to an opened file."""

    var handle: DTypePointer[DType.invalid]
    """The underlying pointer to the file handle."""

    fn __init__(inout self):
        """Default constructor."""
        self.handle = DTypePointer[DType.invalid]()

    fn __init__(inout self, path: String, mode: String) raises:
        """Construct the FileHandle using the file path and mode.

        Args:
          path: The file path.
          mode: The mode to open the file in (the mode can be "r" or "w").
        """
        self.__init__(path._strref_dangerous(), mode._strref_dangerous())

        _ = path
        _ = mode

    fn __init__(inout self, path: StringRef, mode: StringRef) raises:
        """Construct the FileHandle using the file path and string.

        Args:
          path: The file path.
          mode: The mode to open the file in (the mode can be "r" or "w").
        """
        var err_msg = _OwnedStringRef()
        var handle = external_call[
            "KGEN_CompilerRT_IO_FileOpen", DTypePointer[DType.invalid]
        ](path, mode, UnsafePointer.address_of(err_msg))

        if err_msg:
            self.handle = DTypePointer[DType.invalid]()
            raise (err_msg^).consume_as_error()

        self.handle = handle

    @always_inline
    fn __del__(owned self):
        """Closes the file handle."""
        try:
            self.close()
        except:
            pass

    fn close(inout self) raises:
        """Closes the file handle."""
        if self.handle == DTypePointer[DType.invalid]():
            return

        var err_msg = _OwnedStringRef()
        external_call["KGEN_CompilerRT_IO_FileClose", NoneType](
            self.handle, UnsafePointer.address_of(err_msg)
        )

        if err_msg:
            raise (err_msg^).consume_as_error()

        self.handle = DTypePointer[DType.invalid]()

    fn __moveinit__(inout self, owned existing: Self):
        """Moves constructor for the file handle.

        Args:
          existing: The existing file handle.
        """
        self.handle = existing.handle
        existing.handle = DTypePointer[DType.invalid]()

    @always_inline
    fn read(self, size: Int64 = -1) raises -> String:
        """Reads the data from the file.

        Args:
            size: Requested number of bytes to read.

        Returns:
          The contents of the file.
        """
        if self.handle == DTypePointer[DType.invalid]():
            raise Error("invalid file handle")

        var size_copy: Int64 = size
        var err_msg = _OwnedStringRef()

        var buf = external_call[
            "KGEN_CompilerRT_IO_FileRead", UnsafePointer[Int8]
        ](
            self.handle,
            UnsafePointer.address_of(size_copy),
            UnsafePointer.address_of(err_msg),
        )

        if err_msg:
            raise (err_msg^).consume_as_error()

        return String(buf, int(size_copy) + 1)

    fn read_bytes(self, size: Int64 = -1) raises -> List[Int8]:
        """Read from file buffer until we have `size` characters or we hit EOF.
        If `size` is negative or omitted, read until EOF.

        Args:
            size: Requested number of bytes to read.

        Returns:
          The contents of the file.
        """
        if self.handle == DTypePointer[DType.invalid]():
            raise Error("invalid file handle")

        var size_copy: Int64 = size
        var err_msg = _OwnedStringRef()

        var buf = external_call[
            "KGEN_CompilerRT_IO_FileReadBytes", UnsafePointer[Int8]
        ](
            self.handle,
            UnsafePointer.address_of(size_copy),
            UnsafePointer.address_of(err_msg),
        )

        if err_msg:
            raise (err_msg^).consume_as_error()

        var list = List[Int8](capacity=int(size_copy))
        var list_ptr = UnsafePointer[Int8](address=int(list.data))

        # Initialize the List elements and set the initialized size
        memcpy(list_ptr, buf, int(size_copy))
        list.size = int(size_copy)

        return list

    fn seek(self, offset: UInt64, whence: UInt8 = os.SEEK_SET) raises -> UInt64:
        """Seeks to the given offset in the file.

        Args:
            offset: The byte offset to seek to.
            whence: The reference point for the offset:
                os.SEEK_SET = 0: start of file (Default).
                os.SEEK_CUR = 1: current position.
                os.SEEK_END = 2: end of file.

        Raises:
            An error if this file handle is invalid, or if file seek returned a
            failure.

        Returns:
            The resulting byte offset from the start of the file.

        Examples:

        Skip 32 bytes from the current read position:

        ```mojo
        import os
        var f = open("/tmp/example.txt", "r")
        f.seek(os.SEEK_CUR, 32)
        ```

        Start from 32 bytes from the end of the file:

        ```mojo
        import os
        var f = open("/tmp/example.txt", "r")
        f.seek(os.SEEK_END, -32)
        ```
        .
        """
        if not self.handle:
            raise "invalid file handle"

        debug_assert(
            whence >= 0 and whence < 3,
            "Second argument to `seek` must be between 0 and 2.",
        )
        var err_msg = _OwnedStringRef()
        var pos = external_call["KGEN_CompilerRT_IO_FileSeek", UInt64](
            self.handle, offset, whence, UnsafePointer.address_of(err_msg)
        )

        if err_msg:
            raise (err_msg^).consume_as_error()

        return pos

    fn write(self, data: StringLiteral) raises:
        """Write the data to the file.

        Args:
          data: The data to write to the file.
        """
        self.write(StringRef(data))

    fn write(self, data: String) raises:
        """Write the data to the file.

        Args:
          data: The data to write to the file.
        """
        self._write(data._as_ptr(), len(data))

    @always_inline
    fn write(self, data: StringRef) raises:
        """Write the data to the file.

        Args:
          data: The data to write to the file.
        """
        self._write(data.data, len(data))

    @always_inline
    fn _write[
        address_space: AddressSpace
    ](self, ptr: DTypePointer[DType.int8, address_space], len: Int) raises:
        """Write the data to the file.

        Params:
          address_space: The address space of the pointer.

        Args:
          ptr: The pointer to the data to write.
          len: The length of the pointer (in bytes).
        """
        if self.handle == DTypePointer[DType.invalid]():
            raise Error("invalid file handle")

        var err_msg = _OwnedStringRef()
        external_call["KGEN_CompilerRT_IO_FileWrite", NoneType](
            self.handle,
            ptr.address,
            len,
            UnsafePointer.address_of(err_msg),
        )

        if err_msg:
            raise (err_msg^).consume_as_error()

    fn __enter__(owned self) -> Self:
        """The function to call when entering the context."""
        return self^


fn open(path: String, mode: String) raises -> FileHandle:
    """Opens the file specified by path using the mode provided, returning a
    FileHandle.

    Args:
      path: The path to the file to open.
      mode: The mode to open the file in.

    Returns:
      A file handle.
    """
    return FileHandle(path, mode)


fn open[
    pathlike: os.PathLike
](path: pathlike, mode: String) raises -> FileHandle:
    """Opens the file specified by path using the mode provided, returning a
    FileHandle.

    Parameters:
      pathlike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the file to open.
      mode: The mode to open the file in (the mode can be "r" or "w").

    Returns:
      A file handle.
    """
    return FileHandle(path.__fspath__(), mode)
