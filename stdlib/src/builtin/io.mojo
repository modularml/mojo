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
"""Provides utilities for working with input/output.

These are Mojo built-ins, so you don't need to import them.
"""

from sys import (
    bitwidthof,
    external_call,
    stdout,
    triple_is_nvidia_cuda,
    _libc as libc,
)
from sys._libc import dup, fclose, fdopen, fflush
from sys.ffi import OpaquePointer, c_char

from utils import Span, write_buffered, write_args
from collections import InlineArray
from builtin.dtype import _get_dtype_printf_format
from builtin.file_descriptor import FileDescriptor
from memory import UnsafePointer, memcpy

from utils import StringRef, StaticString, StringSlice

# ===----------------------------------------------------------------------=== #
#  _file_handle
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct _fdopen[mode: StringLiteral = "a"]:
    var handle: OpaquePointer

    fn __init__(inout self, stream_id: FileDescriptor):
        """Creates a file handle to the stdout/stderr stream.

        Args:
            stream_id: The stream id
        """

        self.handle = fdopen(dup(stream_id.value), mode.unsafe_cstr_ptr())

    fn __enter__(self) -> Self:
        """Open the file handle for use within a context manager"""
        return self

    fn __exit__(self):
        """Closes the file handle."""
        _ = fclose(self.handle)

    fn readline(self) -> String:
        """Reads an entire line from stdin or until EOF. Lines are delimited by a newline character.

        Returns:
            The line read from the stdin.

        Examples:

        ```mojo
        from builtin.io import _fdopen

        var line = _fdopen["r"](0).readline()
        print(line)
        ```

        Assuming the above program is named `my_program.mojo`, feeding it `Hello, World` via stdin would output:

        ```bash
        echo "Hello, World" | mojo run my_program.mojo

        # Output from print:
        Hello, World
        ```
        .
        """
        return self.read_until_delimiter("\n")

    fn read_until_delimiter(self, delimiter: String) -> String:
        """Reads an entire line from a stream, up to the `delimiter`.
        Does not include the delimiter in the result.

        Args:
            delimiter: The delimiter to read until.

        Returns:
            The text read from the stdin.

        Examples:

        ```mojo
        from builtin.io import _fdopen

        var line = _fdopen["r"](0).read_until_delimiter(",")
        print(line)
        ```

        Assuming the above program is named `my_program.mojo`, feeding it `Hello, World` via stdin would output:

        ```bash
        echo "Hello, World" | mojo run my_program.mojo

        # Output from print:
        Hello
        ```
        """
        # getdelim will allocate the buffer using malloc().
        var buffer = UnsafePointer[UInt8]()
        # ssize_t getdelim(char **restrict lineptr, size_t *restrict n,
        #                  int delimiter, FILE *restrict stream);
        var bytes_read = external_call[
            "getdelim",
            Int,
            UnsafePointer[UnsafePointer[UInt8]],
            UnsafePointer[UInt64],
            Int,
            OpaquePointer,
        ](
            UnsafePointer.address_of(buffer),
            UnsafePointer.address_of(UInt64(0)),
            ord(delimiter),
            self.handle,
        )
        # Copy the buffer (excluding the delimiter itself) into a Mojo String.
        var s = String(StringRef(buffer, bytes_read - 1))
        # Explicitly free the buffer using free() instead of the Mojo allocator.
        libc.free(buffer.bitcast[NoneType]())
        return s


# ===----------------------------------------------------------------------=== #
#  _flush
# ===----------------------------------------------------------------------=== #


@no_inline
fn _flush(file: FileDescriptor = stdout):
    with _fdopen(file) as fd:
        _ = fflush(fd.handle)


# ===----------------------------------------------------------------------=== #
#  _printf
# ===----------------------------------------------------------------------=== #


@no_inline
fn _printf[
    fmt: StringLiteral, *types: AnyType
](*args: *types, file: FileDescriptor = stdout):
    var loaded_pack = args._get_loaded_kgen_pack()
    var f = fmt.unsafe_ptr().bitcast[c_char]()

    @parameter
    if triple_is_nvidia_cuda():
        _ = external_call["vprintf", Int32](f, Pointer.address_of(loaded_pack))
    else:
        with _fdopen(file) as fd:
            # FIXME: external_call should handle this
            _ = __mlir_op.`pop.external_call`[
                func = "KGEN_CompilerRT_fprintf".value,
                variadicType = __mlir_attr[
                    `(`,
                    `!kgen.pointer<none>,`,
                    `!kgen.pointer<scalar<si8>>`,
                    `) -> !pop.scalar<si32>`,
                ],
                _type=Int32,
            ](fd, f, loaded_pack)


# ===----------------------------------------------------------------------=== #
#  _snprintf
# ===----------------------------------------------------------------------=== #


@no_inline
fn _snprintf[
    fmt: StringLiteral, *types: AnyType
](str: UnsafePointer[UInt8], size: Int, *args: *types) -> Int:
    """Writes a format string into an output pointer.

    Parameters:
        fmt: A format string.
        types: The types of arguments interpolated into the format string.

    Args:
        str: A pointer into which the format string is written.
        size: At most, `size - 1` bytes are written into the output string.
        args: Arguments interpolated into the format string.

    Returns:
        The number of bytes written into the output string.
    """

    # FIXME: external_call should handle this
    var f = fmt.unsafe_ptr().bitcast[c_char]()
    var num = __mlir_op.`pop.external_call`[
        func = "snprintf".value,
        variadicType = __mlir_attr[
            `(`,
            `!kgen.pointer<scalar<si8>>,`,
            `!pop.scalar<index>, `,
            `!kgen.pointer<scalar<si8>>`,
            `) -> !pop.scalar<si32>`,
        ],
        _type=Int32,
    ](str.bitcast[c_char](), size, f, args._get_loaded_kgen_pack())
    return int(num)


@no_inline
fn _snprintf_scalar[
    type: DType,
    float_format: StringLiteral = "%.17g",
](buffer: UnsafePointer[UInt8], size: Int, x: Scalar[type]) -> Int:
    @parameter
    if type is DType.bool:
        if x:
            return _snprintf["True"](buffer, size)
        else:
            return _snprintf["False"](buffer, size)
    elif type.is_integral():
        return _snprintf[_get_dtype_printf_format[type]()](buffer, size, x)
    elif (
        type is DType.float16 or type is DType.bfloat16 or type is DType.float32
    ):
        # We need to cast the value to float64 to print it.
        return _float_repr[float_format](buffer, size, x.cast[DType.float64]())
    elif type is DType.float64:
        return _float_repr[float_format](buffer, size, rebind[Float64](x))
    return 0


# ===----------------------------------------------------------------------=== #
#  Helper functions to print a single pop scalar without spacing or new line.
# ===----------------------------------------------------------------------=== #


@no_inline
fn _float_repr[
    fmt: StringLiteral = "%.17g"
](buffer: UnsafePointer[UInt8], size: Int, x: Float64) -> Int:
    # Using `%.17g` with decimal check is equivalent to CPython's fallback path
    # when its more complex dtoa library (forked from
    # https://github.com/dtolnay/dtoa) is not available.
    var n = _snprintf[fmt](buffer, size, x.value)
    # If the buffer isn't big enough to add anything, then just return.
    if n + 2 >= size:
        return n
    # Don't do anything fancy. Just insert ".0" if there is no decimal and this
    # is not in exponent form.
    var p = buffer
    alias minus = ord("-")
    alias dot = ord(".")
    if p[] == minus:
        p += 1
    while p[] != 0 and isdigit(p[]):
        p += 1
    if p[]:
        return n
    p[] = dot
    p += 1
    p[] = ord("0")
    p += 1
    p[] = 0
    return n + 2


# ===----------------------------------------------------------------------=== #
#  _put
# ===----------------------------------------------------------------------=== #


fn _put(strref: StringRef, file: FileDescriptor = stdout):
    var str_slice = StringSlice[ImmutableAnyOrigin](
        unsafe_from_utf8_strref=strref
    )

    _put(str_slice, file=file)


@no_inline
fn _put[
    lif: ImmutableOrigin, //
](x: StringSlice[lif], file: FileDescriptor = stdout):
    # Avoid printing "(null)" for an empty/default constructed `String`
    var str_len = x.byte_length()

    if not str_len:
        return

    @parameter
    if triple_is_nvidia_cuda():
        # Note:
        #   This assumes that the `StringSlice` that was passed in is NUL
        #   terminated.
        var tmp = 0
        var arg_ptr = UnsafePointer.address_of(tmp)
        _ = external_call["vprintf", Int32](
            x.unsafe_ptr(), arg_ptr.bitcast[OpaquePointer]()
        )
    else:
        alias MAX_STR_LEN = 0x1000_0000

        # The string can be printed, so that's fine.
        if str_len < MAX_STR_LEN:
            _printf["%.*s"](x.byte_length(), x.unsafe_ptr(), file=file)
            return

        # The string is large, then we need to chunk it.
        var p = x.unsafe_ptr()
        while str_len:
            var ll = min(str_len, MAX_STR_LEN)
            _printf["%.*s"](ll, p, file=file)
            str_len -= ll
            p += ll


# ===----------------------------------------------------------------------=== #
#  print
# ===----------------------------------------------------------------------=== #


@no_inline
fn print[
    *Ts: Writable
](
    *values: *Ts,
    sep: StaticString = " ",
    end: StaticString = "\n",
    flush: Bool = False,
    owned file: FileDescriptor = stdout,
):
    """Prints elements to the text stream. Each element is separated by `sep`
    and followed by `end`.

    Parameters:
        Ts: The elements types.

    Args:
        values: The elements to print.
        sep: The separator used between elements.
        end: The String to write after printing the elements.
        flush: If set to true, then the stream is forcibly flushed.
        file: The output stream.
    """

    # TODO: Implement a float formatter for GPU to enable buffering to global
    # memory. PTX isn't able to call snprintf to format floats.
    @parameter
    if triple_is_nvidia_cuda():
        write_args(file, values, sep=sep, end=end)
    else:
        write_buffered[buffer_size=4096](file, values, sep=sep, end=end)
        if flush:
            _flush(file=file)


# ===----------------------------------------------------------------------=== #
#  input
# ===----------------------------------------------------------------------=== #


fn input(prompt: String = "") -> String:
    """Reads a line of input from the user.

    Reads a line from standard input, converts it to a string, and returns that string.
    If the prompt argument is present, it is written to standard output without a trailing newline.

    Args:
        prompt: An optional string to be printed before reading input.

    Returns:
        A string containing the line read from the user input.

    Examples:
    ```mojo
    name = input("Enter your name: ")
    print("Hello", name)
    ```

    If the user enters "Mojo" it prints "Hello Mojo".
    """
    if prompt != "":
        print(prompt, end="")
    return _fdopen["r"](0).readline()
