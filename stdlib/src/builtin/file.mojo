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

from memory import AddressSpace, UnsafePointer
from utils.string_slice import _is_newline_start


@register_passable
struct _OwnedStringRef(Boolable):
    var data: UnsafePointer[UInt8]
    var length: Int

    fn __init__() -> _OwnedStringRef:
        return Self {data: UnsafePointer[UInt8](), length: 0}

    fn __del__(owned self):
        if self.data:
            self.data.free()

    fn consume_as_error(owned self) -> Error:
        var data = self.data

        # Don't free self.data in our dtor.
        self.data = UnsafePointer[UInt8]()

        return Error {
            data: data,
            loaded_length: -self.length,
        }

    fn __bool__(self) -> Bool:
        return self.length != 0


@value
struct _FileHandleIter[
    is_mutable: Bool, //,
    lifetime: AnyLifetime[is_mutable].type,
]:
    """Iterator for FileHandle.

    Parameters:
        is_mutable: Whether the slice is mutable.
        lifetime: The lifetime of the underlying data.
    """

    var index: Int
    var continuation_bytes: Int
    var ptr: UnsafePointer[UInt8]
    var length: Int
    var encoding: String

    fn __init__(
        inout self,
        *,
        unsafe_pointer: UnsafePointer[UInt8],
        length: Int,
        encoding: String,
    ):
        self.index = 0 
        self.ptr = unsafe_pointer
        self.length = length
        self.continuation_bytes = 0
        self.encoding = encoding

    fn __iter__(self) -> Self:
        return self

    fn __next__(inout self) -> StringSlice[lifetime]:
        var idx = self.index
        while idx < self.length:
            var read_ahead = 3 if idx < self.length -2 else (
                2 if idx < self.length -1 else 1
            )
            var s = StringSlice[lifetime](
                unsafe_from_utf8_ptr=self.ptr + idx,
                len=read_ahead,
            )
            if self.encoding == "UTF-16":
                s = String.from_utf16(List[UInt16](
                    unsafe_pointer=self.ptr.bitcast[DType.uint16](), 
                    size=self.length // 2, 
                    capacity=self.length // 2,
                )).as_string_slice()
            elif self.encoding == "UTF-32":
                s = String.from_unicode(List[Int](
                    unsafe_pointer=self.ptr.bitcast[DType.index](), 
                    size=self.length // 2, 
                    capacity=self.length // 2,
                )).as_string_slice()
            var res = _is_newline_start(s.unsafe_ptr(), read_ahead):
            if res[0]:
                var val = StringSlice[lifetime](
                    unsafe_from_utf8_ptr=self.ptr + idx + res[1], 
                    len=idx - self.index,
                )
                self.index = idx + res[1]
                return val
            idx += read_ahead

        return StringSlice[lifetime](
            unsafe_from_utf8_ptr=self.ptr + self.index, len=idx - self.index
        )

    fn __len__(self) -> Int:
        return self.length - self.index


struct FileHandle:
    """File handle to an opened file."""

    var handle: UnsafePointer[NoneType]
    """The underlying pointer to the file handle."""
    var _encoding: String

    fn __init__(inout self):
        """Default constructor."""
        self.handle = UnsafePointer[NoneType]()

    fn __init__(
        inout self, path: String, mode: String, encoding: String
    ) raises:
        """Construct the FileHandle using the file path and mode.

        Args:
            path: The file path.
            mode: The mode to open the file in: {"r", "w", "rw"}.
            encoding: The encoding to read the File with.
        """
        self.__init__(
            path._strref_dangerous(), mode._strref_dangerous(), encoding
        )

        _ = path
        _ = mode

    fn __init__(
        inout self, path: StringRef, mode: StringRef, encoding: String
    ) raises:
        """Construct the FileHandle using the file path and string.

        Args:
            path: The file path.
            mode: The mode to open the file in: {"r", "w", "rw"}.
            encoding: The encoding to read the File with.
        """
        var err_msg = _OwnedStringRef()
        var handle = external_call[
            "KGEN_CompilerRT_IO_FileOpen", UnsafePointer[NoneType]
        ](path, mode, Reference(err_msg))

        if err_msg:
            self.handle = UnsafePointer[NoneType]()
            raise err_msg^.consume_as_error()

        self.handle = handle
        self._encoding = encoding

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
            self.handle, Reference(err_msg)
        )

        if err_msg:
            raise err_msg^.consume_as_error()

        self.handle = UnsafePointer[NoneType]()

    fn __moveinit__(inout self, owned existing: Self):
        """Moves constructor for the file handle.

        Args:
          existing: The existing file handle.
        """
        self.handle = existing.handle
        existing.handle = UnsafePointer[NoneType]()

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
            Reference(size_copy),
            Reference(err_msg),
        )

        if err_msg:
            raise err_msg^.consume_as_error()

        return String(buf, int(size_copy) + 1)

    fn read[
        type: DType
    ](self, ptr: UnsafePointer[Scalar[type]], size: Int64 = -1) raises -> Int64:
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
        var ptr = UnsafePointer[Float32].alloc(8)
        var bytes = file.read(ptr, 8)
        print("bytes read", bytes)

        var first_element = ptr[0]
        print(first_element)

        # Skip 2 elements
        _ = file.seek(2 * sizeof[DType.float32](), os.SEEK_CUR)

        # Allocate and load 8 more elements from file handle seek position
        var ptr2 = UnsafePointer[Float32].alloc(8)
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

        var err_msg = _OwnedStringRef()

        var bytes_read = external_call[
            "KGEN_CompilerRT_IO_FileReadToAddress", Int64
        ](
            self.handle,
            ptr,
            size * sizeof[type](),
            Reference(err_msg),
        )

        if err_msg:
            raise (err_msg^).consume_as_error()

        return bytes_read

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

        Reading the entire file into a List[UInt8]:

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
            Reference(size_copy),
            Reference(err_msg),
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
        _ = f.seek(32, os.SEEK_CUR)
        ```

        Start from 32 bytes from the end of the file:

        ```mojo
        import os
        var f = open("/tmp/example.txt", "r")
        _ = f.seek(-32, os.SEEK_END)
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
            self.handle, offset, whence, Reference(err_msg)
        )

        if err_msg:
            raise (err_msg^).consume_as_error()

        return pos

    fn write(self, data: String) raises:
        """Write the data to the file.

        Args:
          data: The data to write to the file.
        """
        self._write(data.unsafe_ptr(), data.byte_length())

    fn write(self, data: Span[UInt8, _]) raises:
        """Write a borrowed sequence of data to the file.

        Args:
          data: The data to write to the file.
        """
        self._write(data.unsafe_ptr(), len(data))

    fn write(self, data: List[UInt8]) raises:
        """Write a borrowed sequence of data to the file.

        Args:
          data: The data to write to the file.
        """
        self._write(data.unsafe_ptr(), len(data))

    fn write(self, data: StringRef) raises:
        """Write the data to the file.

        Args:
          data: The data to write to the file.
        """
        self._write(data.unsafe_ptr(), len(data))

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
            Reference(err_msg),
        )

        if err_msg:
            raise (err_msg^).consume_as_error()

    fn __enter__(owned self) -> Self:
        """The function to call when entering the context.

        Returns:
            The file handle.
        """
        return self^

    fn _get_raw_fd(self) -> Int:
        var i64_res = external_call[
            "KGEN_CompilerRT_IO_GetFD",
            Int64,
        ](self.handle)
        return Int(i64_res.value)

    fn __len__(self) -> Int:
        # TODO: need to fetch file byte size
        return 0

    fn __iter__(ref [_]self) -> _FileHandleIter[__lifetime_of(self)]:
        """Iterate over elements of the string slice, returning immutable references.

        Returns:
            An iterator of references to the string elements.
        """
        return _FileHandleIter[__lifetime_of(self)](
            unsafe_pointer=self.handle, # TODO: need memory mapping
            length=len(self),
            encoding=self._encoding,
        )


fn open[
    PathLike: os.PathLike
](
    path: PathLike, mode: String, encoding: String = "UTF-8"
) raises -> FileHandle:
    """Opens the file specified by path using the mode provided, returning a
    FileHandle.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the file to open.
      mode: The mode to open the file in (the mode can be "r" or "w").
      encoding: The encoding to read the File with.

    Returns:
      A file handle.
    """
    return FileHandle(path.__fspath__(), mode, encoding.upper())
