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

from collections import InlineArray
from sys import _libc as libc
from sys import (
    bitwidthof,
    external_call,
    is_amd_gpu,
    is_gpu,
    is_nvidia_gpu,
    stdout,
)
from sys._libc import dup, fclose, fdopen, fflush
from sys.ffi import OpaquePointer

from builtin.dtype import _get_dtype_printf_format
from builtin.file_descriptor import FileDescriptor
from memory import UnsafePointer, memcpy

from utils import (
    StaticString,
    StringRef,
    StringSlice,
    write_args,
    write_buffered,
)

# ===----------------------------------------------------------------------=== #
#  _file_handle
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct _fdopen[mode: StringLiteral = "a"]:
    var handle: OpaquePointer

    @implicit
    fn __init__(out self, stream_id: FileDescriptor):
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
](*arguments: *types, file: FileDescriptor = stdout):
    # The argument pack will contain references for each value in the pack,
    # but we want to pass their values directly into the C printf call. Load
    # all the members of the pack.
    var loaded_pack = arguments.get_loaded_kgen_pack()

    @parameter
    if is_nvidia_gpu():
        _ = external_call["vprintf", Int32](
            fmt.unsafe_cstr_ptr(), Pointer.address_of(loaded_pack)
        )
    elif is_amd_gpu():
        # constrained[False, "_printf on AMDGPU is not implemented"]()
        pass
    else:
        with _fdopen(file) as fd:
            _ = __mlir_op.`pop.external_call`[
                func = "KGEN_CompilerRT_fprintf".value,
                variadicType = __mlir_attr[
                    `(`,
                    `!kgen.pointer<none>,`,
                    `!kgen.pointer<scalar<si8>>`,
                    `) -> !pop.scalar<si32>`,
                ],
                _type=Int32,
            ](fd, fmt.unsafe_cstr_ptr(), loaded_pack)


# ===----------------------------------------------------------------------=== #
#  _snprintf
# ===----------------------------------------------------------------------=== #


@no_inline
fn _snprintf[
    fmt: StringLiteral, *types: AnyType
](str: UnsafePointer[UInt8], size: Int, *arguments: *types) -> Int:
    """Writes a format string into an output pointer.

    Parameters:
        fmt: A format string.
        types: The types of arguments interpolated into the format string.

    Args:
        str: A pointer into which the format string is written.
        size: At most, `size - 1` bytes are written into the output string.
        arguments: Arguments interpolated into the format string.

    Returns:
        The number of bytes written into the output string.
    """
    # The argument pack will contain references for each value in the pack,
    # but we want to pass their values directly into the C snprintf call. Load
    # all the members of the pack.
    var loaded_pack = arguments.get_loaded_kgen_pack()

    return int(
        __mlir_op.`pop.external_call`[
            func = "snprintf".value,
            variadicType = __mlir_attr[
                `(`,
                `!kgen.pointer<scalar<si8>>,`,
                `!pop.scalar<index>, `,
                `!kgen.pointer<scalar<si8>>`,
                `) -> !pop.scalar<si32>`,
            ],
            _type=Int32,
        ](str, size, fmt.unsafe_cstr_ptr(), loaded_pack)
    )


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

    # TODO(MSTDL-1027): Print on AMD GPUs is not implemented yet.
    @parameter
    if is_amd_gpu():
        return

    write_buffered[buffer_size=4096](file, values, sep=sep, end=end)

    @parameter
    if not is_gpu():
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
