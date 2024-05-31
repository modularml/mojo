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

from memory import AddressSpace, DTypePointer, Pointer


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
        return Error {
            data: UnsafePointer[UInt8]._from_dtype_ptr(
                # TODO: Remove cast once string UInt8 transition is complete.
                data.bitcast[DType.uint8]()
            ),
            loaded_length: -length,
        }

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
          mode: The mode to open the file in (the mode can be "r" or "w" or "rw").
        """
        self.__init__(path._strref_dangerous(), mode._strref_dangerous())

        _ = path
        _ = mode

    fn __init__(inout self, path: StringRef, mode: StringRef) raises:
        """Construct the FileHandle using the file path and string.

        Args:
          path: The file path.
          mode: The mode to open the file in (the mode can be "r" or "w" or "rw").
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
        if not self.handle:
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
        """Reads data from a file and sets the file handle seek position. If
        size is left as the default of -1, it will read to the end of the file.
        Setting size to a number larger than what's in the file will set
        String.size to the total number of bytes, and read all the data.

        Args:
            size: Requested number of bytes to read (Default: -1 = EOF).

        Returns:
          The contents of the file.

        Raises:
            An error if this file handle is invalid, or if the file read
            returned a failure.

        Examples:

        Read the entire file into a String:

        ```mojo
        var file = open("/tmp/example.txt", "r")
        var string = file.read()
        print(string)
        ```

        Read the first 8 bytes, skip 2 bytes, and then read the next 8 bytes:

        ```mojo
        import os
        var file = open("/tmp/example.txt", "r")
        var word1 = file.read(8)
        print(word1)
        _ = file.seek(2, os.SEEK_CUR)
        var word2 = file.read(8)
        print(word2)
        ```

        Read the last 8 bytes in the file, then the first 8 bytes
        ```mojo
        _ = file.seek(-8, os.SEEK_END)
        var last_word = file.read(8)
        print(last_word)
        _ = file.seek(8, os.SEEK_SET) # os.SEEK_SET is the default start of file
        var first_word = file.read(8)
        print(first_word)
        ```
        .
        """
        if not self.handle:
            raise Error("invalid file handle")

        var size_copy: Int64 = size
        var err_msg = _OwnedStringRef()

        var buf = external_call[
            "KGEN_CompilerRT_IO_FileRead", UnsafePointer[UInt8]
        ](
            self.handle,
            UnsafePointer.address_of(size_copy),
            UnsafePointer.address_of(err_msg),
        )

        if err_msg:
            raise (err_msg^).consume_as_error()

        return String(buf, int(size_copy) + 1)

    @always_inline
    fn read[
        type: DType
    ](self, ptr: DTypePointer[type], size: Int64 = -1) raises -> Int64:
        """Read data from the file into the pointer. Setting size will read up
        to `sizeof(type) * size`. The default value of `size` is -1 which
        will read to the end of the file. Starts reading from the file handle
        seek pointer, and after reading adds `sizeof(type) * size` bytes to the
        seek pointer.

        Parameters:
            type: The type that will the data will be represented as.

        Args:
            ptr: The pointer where the data will be read to.
            size: Requested number of elements to read.

        Returns:
            The total amount of data that was read in bytes.

        Raises:
            An error if this file handle is invalid, or if the file read
            returned a failure.

        Examples:

        ```mojo
        import os

        alias file_name = "/tmp/example.txt"
        var file = open(file_name, "r")

        # Allocate and load 8 elements
        var ptr = DTypePointer[DType.float32].alloc(8)
        var bytes = file.read(ptr, 8)
        print("bytes read", bytes)

        var first_element = ptr.load(0)
        print(first_element)

        # Skip 2 elements
        _ = file.seek(2 * sizeof[DType.float32](), os.SEEK_CUR)

        # Allocate and load 8 more elements from file handle seek position
        var ptr2 = DTypePointer[DType.float32].alloc(8)
        var bytes2 = file.read(ptr2, 8)

        var eleventh_element = ptr2[0]
        var twelvth_element = ptr2[1]
        print(eleventh_element, twelvth_element)

        # Free the memory
        ptr.free()
        ptr2.free()
        ```
        .
        """

        if not self.handle:
            raise Error("invalid file handle")

        var size_copy = size * sizeof[type]()
        var err_msg = _OwnedStringRef()

        external_call["KGEN_CompilerRT_IO_FileReadToAddress", NoneType](
            self.handle,
            ptr,
            UnsafePointer.address_of(size_copy),
            UnsafePointer.address_of(err_msg),
        )

        if err_msg:
            raise (err_msg^).consume_as_error()
        return size_copy

    fn read_bytes(self, size: Int64 = -1) raises -> List[UInt8]:
        """Reads data from a file and sets the file handle seek position. If
        size is left as default of -1, it will read to the end of the file.
        Setting size to a number larger than what's in the file will be handled
        and set the List.size to the total number of bytes in the file.

        Args:
            size: Requested number of bytes to read (Default: -1 = EOF).

        Returns:
            The contents of the file.

        Raises:
            An error if this file handle is invalid, or if the file read
            returned a failure.

        Examples:

        Reading the entire file into a List[Int8]:

        ```mojo
        var file = open("/tmp/example.txt", "r")
        var string = file.read_bytes()
        ```

        Reading the first 8 bytes, skipping 2 bytes, and then reading the next
        8 bytes:

        ```mojo
        import os
        var file = open("/tmp/example.txt", "r")
        var list1 = file.read(8)
        _ = file.seek(2, os.SEEK_CUR)
        var list2 = file.read(8)
        ```

        Reading the last 8 bytes in the file, then the first 8 bytes:

        ```mojo
        import os
        var file = open("/tmp/example.txt", "r")
        _ = file.seek(-8, os.SEEK_END)
        var last_data = file.read(8)
        _ = file.seek(8, os.SEEK_SET) # os.SEEK_SET is the default start of file
        var first_data = file.read(8)
        ```
        .
        """
        if not self.handle:
            raise Error("invalid file handle")

        var size_copy: Int64 = size
        var err_msg = _OwnedStringRef()

        var buf = external_call[
            "KGEN_CompilerRT_IO_FileReadBytes", UnsafePointer[UInt8]
        ](
            self.handle,
            UnsafePointer.address_of(size_copy),
            UnsafePointer.address_of(err_msg),
        )

        if err_msg:
            raise (err_msg^).consume_as_error()

        var list = List[UInt8](
            unsafe_pointer=buf, size=int(size_copy), capacity=int(size_copy)
        )

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

    fn write(self, data: String) raises:
        """Write the data to the file.

        Args:
          data: The data to write to the file.
        """
        self._write(data.unsafe_ptr(), len(data))

    @always_inline
    fn write(self, data: StringRef) raises:
        """Write the data to the file.

        Args:
          data: The data to write to the file.
        """
        self._write(data.unsafe_ptr(), len(data))

    @always_inline
    fn _write[
        address_space: AddressSpace
    ](self, ptr: UnsafePointer[UInt8, address_space], len: Int) raises:
        """Write the data to the file.

        Params:
          address_space: The address space of the pointer.

        Args:
          ptr: The pointer to the data to write.
          len: The length of the pointer (in bytes).
        """
        if not self.handle:
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

    fn _get_raw_fd(self) -> Int:
        var i64_res = external_call[
            "KGEN_CompilerRT_IO_GetFD",
            Int64,
        ](self.handle)
        return Int(i64_res.value)


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
