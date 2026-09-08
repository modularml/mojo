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
"""Provides APIs to read and write files.

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

from std.format._utils import _FlushingWriteBuffer
from std.os import PathLike as stdPathLike, abort, makedirs, remove
from std.os import SEEK_SET, SEEK_END
from std.os.path import dirname
from std.ffi import c_int, c_ssize_t, external_call
from std.sys import size_of
from std.sys._libc_errno import ErrNo, get_errno
from std.sys.info import platform_map

from std.collections import Span

# ===----------------------------------------------------------------------=== #
# open() syscall flags
# ===----------------------------------------------------------------------=== #

# File access modes
comptime O_RDONLY = 0x0000
"""Open file for reading only."""
comptime O_WRONLY = 0x0001
"""Open file for writing only."""
comptime O_RDWR = 0x0002
"""Open file for reading and writing."""

# File creation flags
comptime O_CREAT = platform_map[T=Int, "O_CREAT", linux=0x0040, macos=0x0200]()
"""Create file if it doesn't exist."""

comptime O_TRUNC = platform_map[T=Int, "O_TRUNC", linux=0x0200, macos=0x0400]()
"""Truncate file to zero length."""

comptime O_APPEND = platform_map[
    T=Int, "O_APPEND", linux=0x0400, macos=0x0008
]()
"""Append mode: writes always go to end of file."""

comptime O_CLOEXEC = platform_map[
    T=Int, "O_CLOEXEC", linux=0x80000, macos=0x1000000
]()
"""Close file descriptor on exec."""

# ===----------------------------------------------------------------------=== #
# Helper functions
# ===----------------------------------------------------------------------=== #


def _open_file(path: String, mode: String) raises -> Int:
    """Open a file and return its file descriptor.

    This function implements the complex logic for opening files with proper
    handling of modes, directory creation, and special files.

    Args:
        path: The file path to open.
        mode: The file access mode ("r", "w", "rw", or "a").

    Returns:
        The file descriptor of the opened file.

    Raises:
        Error if the file cannot be opened or if the mode is invalid.
    """
    # Parse mode
    var flags: Int
    var create_dirs = False

    if mode == "r":
        flags = O_RDONLY | O_CLOEXEC
    elif mode == "w":
        flags = O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC
        create_dirs = True
    elif mode == "rw":
        flags = O_RDWR | O_CREAT | O_CLOEXEC
        create_dirs = True
    elif mode == "a":
        flags = O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC
        create_dirs = True
    else:
        raise Error(
            'invalid mode: "'
            + mode
            + '". Can only be one of: {"r", "w", "rw", "a"}'
        )

    # Create parent directories if needed
    if create_dirs:
        var parent = dirname(path)
        if parent:
            try:
                makedirs(parent, exist_ok=True)
            except e:
                raise Error(
                    "unable to create directories '"
                    + parent
                    + "': "
                    + String(e)
                )

    # int open(const char *path, int oflag, ...);
    # Mode 0o666 allows read/write for owner, group, and others (modified by
    # umask).
    var path_str = path
    var fd = external_call["open", c_int, num_fixed_args=2](
        path_str.as_c_string_slice(), c_int(flags), c_int(0o666)
    )

    if fd < 0:
        var err = get_errno()
        raise Error("Failed to open file '" + path + "': " + String(err))

    return Int(fd)


# ===----------------------------------------------------------------------=== #
# FileHandle
# ===----------------------------------------------------------------------=== #


struct FileHandle(Defaultable, Movable, Writer):
    """File handle to an opened file."""

    var handle: Int
    """The underlying file descriptor (Unix fd)."""

    def __init__(out self):
        """Default constructor."""
        self.handle = -1

    def __init__(out self, path: StringSlice, mode: StringSlice) raises:
        """Construct the FileHandle using the file path and mode.

        Args:
            path: The file path.
            mode: The mode to open the file in: {"r", "w", "rw", "a"}.

        Raises:
            If file open mode is not one of the supported modes.
            If there is an error when opening the file.
        """
        self.handle = _open_file(String(path), String(mode))

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
        """
        if self.handle < 0:
            return

        var result = external_call["close", c_int](c_int(self.handle))

        if result < 0:
            var err = get_errno()
            # Still mark as closed even on error
            self.handle = -1
            raise Error("Failed to close file: " + String(err))

        self.handle = -1

    def read(self, size: Int = -1) raises -> String:
        """Reads data from a file and sets the file handle seek position. If
        size is left as the default of -1, it will read to the end of the file.
        Setting size to a number larger than what's in the file will set
        the String length to the total number of bytes, and read all the data.

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
        from std.os import SEEK_CUR
        var file = open("/tmp/example.txt", "r")
        var word1 = file.read(8)
        print(word1)
        _ = file.seek(2, SEEK_CUR)
        var word2 = file.read(8)
        print(word2)
        ```

        Read the last 8 bytes in the file, then the first 8 bytes
        ```mojo
        from std.os import SEEK_SET, SEEK_END
        var file = open("/tmp/example.txt", "r")
        _ = file.seek(-8, SEEK_END)
        var last_word = file.read(8)
        print(last_word)
        _ = file.seek(8, SEEK_SET) # SEEK_SET is the default start of file
        var first_word = file.read(8)
        print(first_word)
        ```
        """

        var bytes = self.read_bytes(size)
        return String(from_utf8=bytes)

    def read[
        dtype: DType, origin: MutOrigin
    ](self, buffer: Span[Scalar[dtype], origin]) raises -> Int:
        """Read data from the file into the Span.

        This will read n bytes from the file into the input Span where
        `0 <= n <= len(buffer)`.

        0 is returned when the file is at EOF, or a 0-sized buffer is
        passed in.

        Parameters:
            dtype: The type that the data will be represented as.
            origin: The origin of the passed in Span.

        Args:
            buffer: The mutable Span to read data into.

        Returns:
            The total amount of data that was read in bytes.

        Raises:
            An error if this file handle is invalid, or if the file read
            returned a failure.

        Examples:

        ```mojo
        from std.os import SEEK_CUR
        from std.sys.info import size_of

        comptime file_name = "/tmp/example.txt"
        var file = open(file_name, "r")

        # Allocate and load 8 elements
        var buffer = Array[Float32, length=8](fill=0)
        var bytes = file.read(buffer)
        print("bytes read", bytes)

        var first_element = buffer[0]
        print(first_element)

        # Skip 2 elements
        _ = file.seek(2 * size_of[DType.float32](), SEEK_CUR)

        # Allocate and load 8 more elements from file handle seek position
        var buffer2 = Array[Float32, length=8](fill=0)
        var bytes2 = file.read(buffer2)

        var eleventh_element = buffer2[0]
        var twelvth_element = buffer2[1]
        print(eleventh_element, twelvth_element)
        ```
        """

        if self.handle < 0:
            raise Error("invalid file handle")

        var fd = self._get_raw_fd()
        var bytes_read = external_call["read", c_ssize_t](
            fd,
            buffer.unsafe_ptr(),
            len(buffer) * size_of[dtype](),
        )

        if bytes_read < 0:
            var err = get_errno()
            raise Error("Failed to read from file: " + String(err))

        return bytes_read

    def read_bytes(self, size: Int = -1) raises -> List[UInt8]:
        """Reads data from a file and sets the file handle seek position. If
        size is left as default of -1, it will read to the end of the file.
        Setting size to a number larger than what's in the file will be handled
        and set the List length to the total number of bytes in the file.

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
        from std.os import SEEK_CUR
        var file = open("/tmp/example.txt", "r")
        var list1 = file.read(8)
        _ = file.seek(2, SEEK_CUR)
        var list2 = file.read(8)
        ```

        Reading the last 8 bytes in the file, then the first 8 bytes:

        ```mojo
        from std.os import SEEK_SET, SEEK_END
        var file = open("/tmp/example.txt", "r")
        _ = file.seek(-8, SEEK_END)
        var last_data = file.read(8)
        _ = file.seek(8, SEEK_SET) # SEEK_SET is the default start of file
        var first_data = file.read(8)
        ```
        """
        if self.handle < 0:
            raise Error("invalid file handle")

        # Start out with the correct size if we know it, otherwise use 256.
        var result = List[UInt8](
            unsafe_uninit_length=size if size >= 0 else 256
        )

        var fd = self._get_raw_fd()
        var num_read = 0
        while True:
            # Read bytes into the list buffer and get the number of bytes
            # successfully read. This may return with a partial read, and
            # signifies EOF with a result of zero bytes.
            var chunk_bytes_to_read = len(result) - num_read
            var chunk_bytes_read = external_call["read", c_ssize_t](
                fd,
                result.unsafe_ptr().unsafe_offset(num_read),
                chunk_bytes_to_read,
            )

            if chunk_bytes_read < 0:
                var err = get_errno()
                raise Error("Failed to read from file: " + String(err))

            num_read += chunk_bytes_read

            # If we read all of the 'size' bytes then we're done.
            if num_read == size or chunk_bytes_read == 0:
                result.shrink(num_read)  # Trim off any tail.
                break

            # If we are reading to EOF, keep reading the next chunk, taking
            # bigger bites each time.
            if size < 0:
                result.resize(unsafe_uninit_length=num_read * 2)

        return result^

    def seek(self, offset: Int, whence: UInt8 = SEEK_SET) raises -> UInt64:
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

        Examples:

        Skip 32 bytes from the current read position:

        ```mojo
        from std.os import SEEK_CUR
        var f = open("/tmp/example.txt", "r")
        _ = f.seek(32, SEEK_CUR)
        ```

        Start from 32 bytes from the end of the file:

        ```mojo
        from std.os import SEEK_END
        var f = open("/tmp/example.txt", "r")
        _ = f.seek(-32, SEEK_END)
        ```
        """
        if self.handle < 0:
            raise "invalid file handle"

        assert (
            whence >= 0 and whence < 3
        ), "Second argument to `seek` must be between 0 and 2."

        var fd = self._get_raw_fd()
        # lseek returns off_t which is typically Int64 on Unix systems
        var pos = external_call["lseek", Int64](fd, Int64(offset), Int(whence))

        if pos < 0:
            var err = get_errno()
            raise Error("Failed to seek in file: " + String(err))

        return UInt64(pos)

    def write_once(mut self, bytes: Span[Byte, _]) raises -> Int:
        """Attempt to write bytes to the file, returning the number of bytes written.

        This is a low-level method that performs a single write syscall. It may
        write fewer bytes than requested (partial write), which is not an error.
        Most users should use `write_bytes()` instead, which handles partial
        writes automatically.

        Args:
            bytes: The byte span to write to this file.

        Returns:
            The number of bytes actually written. This may be less than `len(bytes)`.

        Raises:
            If the file handle is invalid or the write syscall fails.

        Notes:
            Similar to Rust's `Write::write()`, this method represents one attempt
            to write data and may not write all bytes. Use `write_bytes()` for
            guaranteed complete writes (equivalent to Rust's `write_all()`).

        Examples:

        ```mojo
        var file = open("/tmp/example.txt", "w")
        var data = String("Hello, World!").as_bytes()

        # May write fewer bytes than requested
        var bytes_written = file.write_once(data)
        print("Wrote", bytes_written, "of", len(data), "bytes")
        ```
        """
        if self.handle < 0:
            raise Error("invalid file handle")

        var fd = self._get_raw_fd()
        var bytes_written = external_call["write", c_ssize_t](
            fd, bytes.unsafe_ptr(), len(bytes)
        )

        if bytes_written < 0:
            var err = get_errno()
            raise Error("Failed to write to file: " + String(err))

        return bytes_written

    def write_all(mut self, bytes: Span[Byte, _]) raises:
        """Write all bytes to the file, handling partial writes automatically.

        This method guarantees that all bytes are written by looping until
        complete or an error occurs. This is equivalent to Rust's `write_all()`.

        Args:
            bytes: The byte span to write to this file.

        Raises:
            If the file handle is invalid or the write operation fails.

        Notes:
            Unlike `write_once()`, this method will not return until all bytes
            are written or an unrecoverable error occurs. Use this method when
            you need proper error handling for write operations.

        Examples:

        ```mojo
        var file = open("/tmp/example.txt", "w")
        var data = String("Hello, World!").as_bytes()

        # Writes all bytes, handling partial writes automatically
        file.write_all(data)
        ```
        """
        if self.handle < 0:
            raise Error("invalid file handle")

        var total_written = 0
        while total_written < len(bytes):
            var chunk_written = self.write_once(bytes[total_written:])

            # write() returning 0 typically means the object cannot accept more bytes
            if chunk_written == 0:
                raise Error(
                    "Write returned 0 bytes (file may be full or closed)"
                )

            total_written += chunk_written

    @always_inline
    def write_bytes(mut self, bytes: Span[Byte, _]):
        """Write a span of bytes to the file.

        This method is required by the Writer trait and handles partial writes
        automatically by looping until all bytes are written. On write failure,
        the program will abort. For better error handling, use `write_all()`.

        Args:
            bytes: The byte span to write to this file.

        Notes:
            This method satisfies the Writer trait requirement. On write failure,
            the program will abort. Use `write_all()` for proper error handling.

            We use abort() instead of raising errors because the Writer trait
            currently requires write_bytes() to be non-raising. This allows
            the trait to work with both infallible writers (String) and fallible
            writers (files). A future improvement would be to make the Writer
            trait allow raises, enabling proper error handling here.

        Examples:

        ```mojo
        var file = open("/tmp/example.txt", "w")
        var data = String("Hello, World!").as_bytes()
        file.write_bytes(data)  # Aborts on error - use write_all() instead
        ```
        """
        # NOTE: We cannot raise here because Writer trait requires non-raising.
        # This is a design limitation that should be addressed by updating the
        # trait to allow raises in the future.
        if self.handle < 0:
            abort("invalid file handle in write_bytes()")

        var total_written = 0
        while total_written < len(bytes):
            var fd = self._get_raw_fd()
            var bytes_written = external_call["write", c_ssize_t](
                fd,
                bytes.unsafe_ptr().unsafe_offset(total_written),
                len(bytes) - total_written,
            )

            if bytes_written < 0:
                abort("write() syscall failed")

            if bytes_written == 0:
                abort("write() returned 0 bytes (file may be full or closed)")

            total_written += bytes_written

    def write_string(mut self, string: StringSlice):
        """
        Write a `StringSlice` to this `FileHandle`.

        This method is required by the `Writer` trait.

        Args:
            string: The `StringSlice` to write to this `FileHandle`.
        """
        self.write_bytes(string.as_bytes())

    def write[*Ts: Writable](mut self, *args: *Ts):
        """Write a sequence of Writable arguments to the provided Writer.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this Writer.

        Notes:
            Passing an invalid file handle (e.g., after calling `close()`) is
            undefined behavior. In debug builds, this will trigger an assertion.
        """
        assert self.handle >= 0, "invalid file handle in write()"

        var file = FileDescriptor(self._get_raw_fd())
        var buffer = _FlushingWriteBuffer(file)

        comptime for i in range(args.__len__()):
            args[i].write_to(buffer)

        buffer.flush()

    def _write(
        self,
        ptr: ImmPointer[UInt8, _, address_space=_],
        len: Int,
    ) raises:
        """Write the data to the file, handling partial writes automatically.

        Args:
          ptr: The pointer to the data to write.
          len: The length of the data buffer (in bytes).

        Raises:
            If the file handle is invalid or the write operation fails.
        """
        if self.handle < 0:
            raise Error("invalid file handle")

        var fd = self._get_raw_fd()
        var total_written = 0

        while total_written < len:
            var current_ptr = ptr.unsafe_offset(total_written)
            var bytes_written = external_call["write", c_ssize_t](
                fd, current_ptr, len - total_written
            )

            if bytes_written < 0:
                var err = get_errno()
                raise Error("Failed to write to file: " + String(err))

            # write() returning 0 typically means the object cannot accept more bytes
            if bytes_written == 0:
                raise Error(
                    "Write returned 0 bytes (file may be full or closed)"
                )

            total_written += bytes_written

    def __enter__(var self) -> Self:
        """The function to call when entering the context.

        Returns:
            The file handle.
        """
        return self^

    def _get_raw_fd(self) -> Int:
        """Get the raw Unix file descriptor.

        Returns:
            The file descriptor as an Int.
        """
        return self.handle


def open[
    PathLike: stdPathLike
](path: PathLike, mode: StringSlice) raises -> FileHandle:
    """Opens the file specified by path using the mode provided, returning a
    FileHandle.

    Parameters:
        PathLike: The a type conforming to the os.PathLike trait.

    Args:
        path: The path to the file to open.
        mode: The mode to open the file in: {"r", "w", "rw", "a"}.

    Returns:
        A file handle.

    Raises:
        If file open mode is not one of the supported modes.
        If there is an error when opening the file.
    """
    return FileHandle(path.__fspath__(), mode)
