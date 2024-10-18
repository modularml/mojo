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
from memory import memcpy, UnsafePointer


# ===----------------------------------------------------------------------===#


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
    from utils import Span

    @value
    struct NewString(Writer):
        var s: String

        # Enable a type to write its data members as a `Byte[Span]`
        fn write_bytes(inout self, bytes: Span[Byte, _]):
            # If your Writer needs to store the number of bytes being written,
            # you can use e.g. `self.bytes_written += len(str_slice)` here.
            self.s._iadd[False](bytes)

        # Enable passing multiple args that implement `write_to`, which
        # themselves may have calls to `write` with variadic args:
        fn write[*Ts: Writable](inout self, *args: *Ts):
            # Loop through the args, running all their `write_to` functions:
            @parameter
            fn write_arg[T: Writable](arg: T):
                arg.write_to(self)
            args.each[write_arg]()

    @value
    struct Point(Writable):
        var x: Int
        var y: Int

        fn write_to[W: Writer](self, inout writer: W):
            # Write a single `Span[Byte]`:
            var string = "Point"
            writer.write_bytes(string.as_bytes())

            # Write the Ints and StringLiterals in a single call, they also
            # implement `write_to`, which implements how to write themselves as
            # a `Byte[Span]`:
            writer.write("(", self.x, ", ", self.y, ")")

    var output = NewString(String())
    var point = Point(2, 4)

    output.write(point)
    print(output.s)
    ```

    Output:

    ```plaintext
    Point(2, 4)
    ```
    """

    @always_inline
    fn write_bytes(inout self, bytes: Span[Byte, _]):
        """
        Write a `Span[Byte]` to this `Writer`.

        Args:
            bytes: The string slice to write to this Writer. Must NOT be
              null-terminated.
        """
        ...

    fn write[*Ts: Writable](inout self, *args: *Ts):
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


# ===----------------------------------------------------------------------===#
# Writable
# ===----------------------------------------------------------------------===#


trait Writable:
    """The `Writable` trait describes how a type is written into a `Writer`.

    You must implement `write_to` which takes `self` and a type conforming to
    `Writer`:

    ```mojo
    struct Point(Writable):
        var x: Float64
        var y: Float64

        fn write_to[W: Writer](self, inout writer: W):
            var string = "Point"
            # Write a single `Span[Byte]`:
            writer.write_bytes(string.as_bytes())
            # Pass multiple args that can be converted to a `Span[Byte]`:
            writer.write("(", self.x, ", ", self.y, ")")
    ```
    """

    fn write_to[W: Writer](self, inout writer: W):
        """
        Formats the string representation of this type to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The type conforming to `Writable`.
        """
        ...


# ===----------------------------------------------------------------------===#
# Utils
# ===----------------------------------------------------------------------===#


fn write_args[
    W: Writer, *Ts: Writable
](
    inout writer: W,
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


struct _WriteBuffer[W: MovableWriter, //, capacity: Int](Writer):
    var data: InlineArray[UInt8, capacity]
    var pos: Int
    var writer: W

    fn __init__(inout self, owned writer: W):
        self.data = InlineArray[UInt8, capacity](unsafe_uninitialized=True)
        self.pos = 0
        self.writer = writer^

    fn flush(inout self):
        self.writer.write_bytes(
            Span[Byte, ImmutableAnyOrigin](
                unsafe_ptr=self.data.unsafe_ptr(), len=self.pos
            )
        )
        self.pos = 0

    fn write_bytes(inout self, bytes: Span[Byte, _]):
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

    fn write[*Ts: Writable](inout self, *args: *Ts):
        @parameter
        fn write_arg[T: Writable](arg: T):
            arg.write_to(self)

        args.each[write_arg]()


fn write_buffered[
    buffer_size: Int, W: MovableWriter, *Ts: Writable
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
        buffer_size: How many bytes to write to a buffer before writing out.
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
    var buffer = _WriteBuffer[buffer_size](writer^)
    write_args(buffer, args, sep=sep, end=end)
    buffer.flush()
