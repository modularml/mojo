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
"""Establishes the contract between `Writer` and `Writable` types."""

from collections import InlineArray
from sys.info import is_gpu

from memory import UnsafePointer, memcpy, Span

from utils import StaticString

# ===-----------------------------------------------------------------------===#


trait Writer:
    """Describes a type that can be written to by any type that implements the
    `write_to` function.

    This enables you to write one implementation that can be written to a
    variety of types such as file descriptors, strings, network locations etc.
    The types are written as a `Span[Byte]`, so the `Writer` can avoid
    allocations depending on the requirements. There is also a general `write`
    that takes multiple args that implement `write_to`.

    Example:

    ```mojo
    from memory import Span

    @value
    struct NewString(Writer, Writable):
        var s: String

        # Writer requirement to write a Span of Bytes
        fn write_bytes(mut self, bytes: Span[Byte, _]):
            self.s._iadd[False](bytes)

        # Writer requirement to take multiple args
        fn write[*Ts: Writable](mut self, *args: *Ts):
            @parameter
            fn write_arg[T: Writable](arg: T):
                arg.write_to(self)

            args.each[write_arg]()

        # Also make it Writable to allow `print` to write the inner String
        fn write_to[W: Writer](self, mut writer: W):
            writer.write(self.s)


    @value
    struct Point(Writable):
        var x: Int
        var y: Int

        # Pass multiple args to the Writer. The Int and StringLiteral types
        # call `writer.write_bytes` in their own `write_to` implementations.
        fn write_to[W: Writer](self, mut writer: W):
            writer.write("Point(", self.x, ", ", self.y, ")")

        # Enable conversion to a String using `str(point)`
        fn __str__(self) -> String:
            return String.write(self)


    fn main():
        var point = Point(1, 2)
        var new_string = NewString(str(point))
        new_string.write("\\n", Point(3, 4))
        print(new_string)
    ```

    Output:

    ```plaintext
    Point(1, 2)
    Point(3, 4)
    ```
    """

    @always_inline
    fn write_bytes(mut self, bytes: Span[Byte, _]):
        """
        Write a `Span[Byte]` to this `Writer`.

        Args:
            bytes: The string slice to write to this Writer. Must NOT be
              null-terminated.
        """
        ...

    fn write[*Ts: Writable](mut self, *args: *Ts):
        """Write a sequence of Writable arguments to the provided Writer.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this Writer.
        """
        ...
        # TODO: When have default implementations on traits, we can use this:
        # @parameter
        # fn write_arg[W: Writable](arg: W):
        #     arg.write_to(self)
        # args.each[write_arg]()
        #
        # To only have to implement `write_bytes` to make a type a valid Writer


# ===-----------------------------------------------------------------------===#
# Writable
# ===-----------------------------------------------------------------------===#


trait Writable:
    """The `Writable` trait describes how a type is written into a `Writer`.

    You must implement `write_to` which takes `self` and a type conforming to
    `Writer`:

    ```mojo
    struct Point(Writable):
        var x: Float64
        var y: Float64

        fn write_to[W: Writer](self, mut writer: W):
            var string = "Point"
            # Write a single `Span[Byte]`:
            writer.write_bytes(string.as_bytes())
            # Pass multiple args that can be converted to a `Span[Byte]`:
            writer.write("(", self.x, ", ", self.y, ")")
    ```
    """

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats the string representation of this type to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The type conforming to `Writable`.
        """
        ...


# ===-----------------------------------------------------------------------===#
# Utils
# ===-----------------------------------------------------------------------===#


fn write_args[
    W: Writer, *Ts: Writable
](
    mut writer: W,
    args: VariadicPack[_, Writable, *Ts],
    *,
    sep: StaticString = "",
    end: StaticString = "",
):
    """
    Add seperators and end characters when writing variadics into a `Writer`.

    Parameters:
        W: The type of the `Writer` to write to.
        Ts: The types of each arg to write. Each type must satisfy `Writable`.

    Args:
        writer: The `Writer` to write to.
        args: A VariadicPack of Writable arguments.
        sep: The separator used between elements.
        end: The String to write after printing the elements.

    Example

    ```mojo
    import sys
    from utils import write_args

    fn variadic_pack_function[*Ts: Writable](
        *args: *Ts, sep: StringLiteral, end: StringLiteral
    ):
        var stdout = sys.stdout
        write_args(stdout, args, sep=sep, end=end)

    variadic_pack_function(3, "total", "args", sep=",", end="[end]")
    ```

    ```
    3, total, args[end]
    ```
    .
    """

    @parameter
    fn print_with_separator[i: Int, T: Writable](value: T):
        value.write_to(writer)

        @parameter
        if i < len(VariadicList(Ts)) - 1:
            sep.write_to(writer)

    args.each_idx[print_with_separator]()
    if end:
        end.write_to(writer)


trait MovableWriter(Movable, Writer):
    """Allows moving a Writer into a buffer."""

    ...


struct _WriteBufferHeap(Writer):
    var data: UnsafePointer[UInt8]
    var pos: Int

    fn __init__(out self, size: Int):
        self.data = UnsafePointer[
            UInt8, address_space = AddressSpace.GENERIC
        ].alloc(size)
        self.pos = 0

    fn __del__(owned self):
        self.data.free()

    @always_inline
    fn write_bytes(mut self, bytes: Span[UInt8, _]):
        len_bytes = len(bytes)
        # If empty then return
        if len_bytes == 0:
            return
        var ptr = bytes.unsafe_ptr()
        for i in range(len_bytes):
            self.data[i + self.pos] = ptr[i]
        self.pos += len_bytes

    fn write[*Ts: Writable](mut self, *args: *Ts):
        @parameter
        fn write_arg[T: Writable](arg: T):
            arg.write_to(self)

        args.each[write_arg]()


struct _ArgBytes(Writer):
    var size: Int

    fn __init__(out self):
        self.size = 0

    fn write_bytes(mut self, bytes: Span[UInt8, _]):
        self.size += len(bytes)

    fn write[*Ts: Writable](mut self, *args: *Ts):
        @parameter
        fn write_arg[T: Writable](arg: T):
            arg.write_to(self)

        args.each[write_arg]()


struct _WriteBufferStack[W: MovableWriter, //, capacity: Int](Writer):
    var data: InlineArray[UInt8, capacity]
    var pos: Int
    var writer: W

    @implicit
    fn __init__(out self, owned writer: W):
        self.data = InlineArray[UInt8, capacity](unsafe_uninitialized=True)
        self.pos = 0
        self.writer = writer^

    fn flush(mut self):
        self.writer.write_bytes(
            Span[Byte, ImmutableAnyOrigin](
                ptr=self.data.unsafe_ptr(), length=self.pos
            )
        )
        self.pos = 0

    fn write_bytes(mut self, bytes: Span[Byte, _]):
        len_bytes = len(bytes)
        # If empty then return
        if len_bytes == 0:
            return
        # If span is too large to fit in buffer, write directly and return
        if len_bytes > capacity:
            self.flush()
            self.writer.write_bytes(bytes)
            return
        # If buffer would overflow, flush writer and reset pos to 0.
        if self.pos + len_bytes > capacity:
            self.flush()
        # Continue writing to buffer
        memcpy(self.data.unsafe_ptr() + self.pos, bytes.unsafe_ptr(), len_bytes)
        self.pos += len_bytes

    fn write[*Ts: Writable](mut self, *args: *Ts):
        @parameter
        fn write_arg[T: Writable](arg: T):
            arg.write_to(self)

        args.each[write_arg]()


fn write_buffered[
    W: MovableWriter, //,
    *Ts: Writable,
    buffer_size: Int,
](
    owned writer: W,
    args: VariadicPack[_, Writable, *Ts],
    *,
    sep: StaticString = "",
    end: StaticString = "",
):
    """
    Use a buffer on the stack to minimize expensive calls to the writer. When
    the buffer would overflow it writes to the `writer` passed in. You can also
    add seperators between the args, and end characters.


    Parameters:
        W: The type of the `Writer` to write to.
        Ts: The types of each arg to write. Each type must satisfy `Writable`.
        buffer_size: How many bytes to write to a buffer before writing out.

    Args:
        writer: The `Writer` to write to.
        args: A VariadicPack of Writable arguments.
        sep: The separator used between elements.
        end: The String to write after printing the elements.

    Example

    ```mojo
    import sys
    from utils import write_buffered

    fn print_err_buffered[*Ts: Writable](
        *args: *Ts, sep: StringLiteral, end: StringLiteral
    ):
        var stdout = sys.stderr
        write_buffered[buffer_size=4096](stdout, args, sep=sep, end=end)

    print_err_buffered(3, "total", "args", sep=",", end="[end]")
    ```

    ```
    3, total, args[end]
    ```
    .
    """

    @parameter
    if is_gpu():
        # Stack space is very small on GPU due to many threads, so use heap
        # Count the total length of bytes to allocate only once
        var arg_bytes = _ArgBytes()
        write_args(arg_bytes, args, sep=sep, end=end)

        var buffer = _WriteBufferHeap(arg_bytes.size + 1)
        write_args(buffer, args, sep=sep, end=end)
        buffer.data[buffer.pos] = 0
        writer.write_bytes(
            Span[Byte, ImmutableAnyOrigin](ptr=buffer.data, length=buffer.pos)
        )
    else:
        var buffer = _WriteBufferStack[buffer_size](writer^)
        write_args(buffer, args, sep=sep, end=end)
        buffer.flush()
