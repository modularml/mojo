# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
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

from pathlib.path import Path
from sys import external_call

from memory.unsafe import DTypePointer, Pointer, AddressSpace
from tensor import Tensor


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
        let data = self.data
        let length = self.length
        __mlir_op.`lit.ownership.mark_destroyed`(__get_ref_from_value(self))
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

    fn __init__(inout self, path: StringLiteral, mode: StringLiteral) raises:
        """Construct the FileHandle using the file path and mode.

        Args:
          path: The file path.
          mode: The mode to open the file in (the mode can be "r" or "w").
        """
        self.__init__(StringRef(path), StringRef(mode))

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
        let handle = external_call[
            "KGEN_CompilerRT_IO_FileOpen", DTypePointer[DType.invalid]
        ](path, mode, Pointer.address_of(err_msg))

        if err_msg:
            self.handle = DTypePointer[DType.invalid]()
            raise (err_msg ^).consume_as_error()

        self.handle = handle

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
            self.handle, Pointer.address_of(err_msg)
        )

        if err_msg:
            raise (err_msg ^).consume_as_error()

        self.handle = DTypePointer[DType.invalid]()

    fn __moveinit__(inout self, owned existing: Self):
        """Moves constructor for the file handle.

        Args:
          existing: The existing file handle.
        """
        self.handle = existing.handle
        existing.handle = DTypePointer[DType.invalid]()

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

        let buf = external_call["KGEN_CompilerRT_IO_FileRead", Pointer[Int8]](
            self.handle,
            Pointer.address_of(size_copy),
            Pointer.address_of(err_msg),
        )

        if err_msg:
            raise (err_msg ^).consume_as_error()

        return String(buf, int(size_copy) + 1)

    fn read_bytes(self, size: Int64 = -1) raises -> Tensor[DType.int8]:
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

        let buf = external_call[
            "KGEN_CompilerRT_IO_FileReadBytes", Pointer[Int8]
        ](
            self.handle,
            Pointer.address_of(size_copy),
            Pointer.address_of(err_msg),
        )

        if err_msg:
            raise (err_msg ^).consume_as_error()

        return Tensor(DTypePointer[DType.int8](buf.address), int(size_copy))

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
        if not self.handle:
            raise "invalid file handle"

        var err_msg = _OwnedStringRef()
        let pos = external_call["KGEN_CompilerRT_IO_FileSeek", UInt64](
            self.handle, offset, Pointer.address_of(err_msg)
        )

        if err_msg:
            raise (err_msg ^).consume_as_error()

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

    fn write(self, data: StringRef) raises:
        """Write the data to the file.

        Args:
          data: The data to write to the file.
        """
        self._write(data.data, len(data))

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
            Pointer.address_of(err_msg),
        )

        if err_msg:
            raise (err_msg ^).consume_as_error()

    fn __enter__(owned self) -> Self:
        """The function to call when entering the context."""
        return self ^


fn open(path: StringLiteral, mode: StringLiteral) raises -> FileHandle:
    """Opens the file specified by path using the mode provided, returning a
    FileHandle.

    Args:
      path: The path to the file to open.
      mode: The mode to open the file in (the mode can be "r" or "w").

    Returns:
      A file handle.
    """
    return FileHandle(StringRef(path), StringRef(mode))


fn open(path: StringRef, mode: StringRef) raises -> FileHandle:
    """Opens the file specified by path using the mode provided, returning a
    FileHandle.

    Args:
      path: The path to the file to open.
      mode: The mode to open the file in (the mode can be "r" or "w").

    Returns:
      A file handle.
    """
    return FileHandle(path, mode)


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


fn open(path: Path, mode: String) raises -> FileHandle:
    """Opens the file specified by path using the mode provided, returning a
    FileHandle.

    Args:
      path: The path to the file to open.
      mode: The mode to open the file in (the mode can be "r" or "w").

    Returns:
      A file handle.
    """
    return FileHandle(str(path), mode)
