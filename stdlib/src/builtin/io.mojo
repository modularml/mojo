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
    os_is_windows,
    stdout,
    triple_is_nvidia_cuda,
)

from builtin.builtin_list import _LITRefPackHelper
from builtin.dtype import _get_dtype_printf_format
from builtin.file_descriptor import FileDescriptor
from memory import UnsafePointer

from utils import StringRef, unroll, StaticString, StringSlice
from utils._format import Formattable, Formatter, write_to

# ===----------------------------------------------------------------------=== #
#  _file_handle
# ===----------------------------------------------------------------------=== #


fn _dup(fd: Int32) -> Int32:
    @parameter
    if os_is_windows():
        return external_call["_dup", Int32](fd)
    else:
        return external_call["dup", Int32](fd)


@value
@register_passable("trivial")
struct _fdopen:
    alias STDOUT = 1
    alias STDERR = 2
    var handle: UnsafePointer[NoneType]

    fn __init__(inout self, stream_id: FileDescriptor):
        """Creates a file handle to the stdout/stderr stream.

        Args:
            stream_id: The stream id
        """
        alias mode = "a"
        var handle: UnsafePointer[NoneType]

        @parameter
        if os_is_windows():
            handle = external_call["_fdopen", UnsafePointer[NoneType]](
                _dup(stream_id.value), mode.unsafe_cstr_ptr()
            )
        else:
            handle = external_call["fdopen", UnsafePointer[NoneType]](
                _dup(stream_id.value), mode.unsafe_cstr_ptr()
            )
        self.handle = handle

    fn __enter__(self) -> Self:
        return self

    fn __exit__(self):
        """Closes the file handle."""
        _ = external_call["fclose", Int32](self.handle)


# ===----------------------------------------------------------------------=== #
#  _flush
# ===----------------------------------------------------------------------=== #


@no_inline
fn _flush(file: FileDescriptor = stdout):
    with _fdopen(file) as fd:
        _ = external_call["fflush", Int32](fd.handle)


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
    var loaded_pack = _LITRefPackHelper(arguments._value).get_loaded_kgen_pack()

    @parameter
    if triple_is_nvidia_cuda():
        _ = external_call["vprintf", Int32](
            fmt.unsafe_cstr_ptr(), Reference(loaded_pack)
        )
        _ = loaded_pack
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
    var loaded_pack = _LITRefPackHelper(arguments._value).get_loaded_kgen_pack()

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


@no_inline
fn _put(x: Int, file: FileDescriptor = stdout):
    """Prints a scalar value.

    Args:
        x: The value to print.
        file: The output stream.
    """
    _printf[_get_dtype_printf_format[DType.index]()](x, file=file)


@no_inline
fn _put_simd_scalar[type: DType](x: Scalar[type]):
    """Prints a scalar value.

    Parameters:
        type: The DType of the value.

    Args:
        x: The value to print.
    """
    alias format = _get_dtype_printf_format[type]()

    @parameter
    if type is DType.bool:
        _put["True"]() if x else _put["False"]()
    elif type.is_integral():
        _printf[format](x)
    elif type.is_floating_point():

        @parameter
        if triple_is_nvidia_cuda():
            _printf[format](x.cast[DType.float64]())
        else:
            _put(str(x).as_string_slice())
    else:
        constrained[False, "invalid dtype"]()


@no_inline
fn _put[type: DType, simd_width: Int](x: SIMD[type, simd_width]):
    """Prints a scalar value.

    Parameters:
        type: The DType of the value.
        simd_width: The SIMD width.

    Args:
        x: The value to print.
    """
    alias format = _get_dtype_printf_format[type]()

    @parameter
    if simd_width == 1:
        _put_simd_scalar(x[0])
    elif type.is_integral():
        _put["["]()

        @parameter
        for i in range(simd_width):
            _put_simd_scalar(x[i])
            if i != simd_width - 1:
                _put[", "]()
        _put["]"]()
    else:
        _put(str(x).as_string_slice())


@no_inline
fn _put[x: StringLiteral](file: FileDescriptor = stdout):
    _put(x.as_string_slice(), file=file)


fn _put(strref: StringRef, file: FileDescriptor = stdout):
    var str_slice = StringSlice[ImmutableStaticLifetime](
        unsafe_from_utf8_strref=strref
    )

    _put(str_slice, file=file)


@no_inline
fn _put(x: DType, file: FileDescriptor = stdout):
    _put(str(x).as_string_slice(), file=file)


# TODO: Constrain to `StringSlice[False, _]`
@no_inline
fn _put(x: StringSlice, file: FileDescriptor = stdout):
    # Avoid printing "(null)" for an empty/default constructed `String`
    var str_len = x._byte_length()

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
            x.unsafe_ptr(), arg_ptr.bitcast[UnsafePointer[NoneType]]()
        )
        _ = tmp
    else:
        alias MAX_STR_LEN = 0x1000_0000

        # The string can be printed, so that's fine.
        if str_len < MAX_STR_LEN:
            _printf["%.*s"](x._byte_length(), x.unsafe_ptr(), file=file)
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
    *Ts: Formattable
](
    *values: *Ts,
    sep: StaticString = " ",
    end: StaticString = "\n",
    flush: Bool = False,
    file: FileDescriptor = stdout,
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

    var writer = Formatter(fd=file)

    @parameter
    fn print_with_separator[i: Int, T: Formattable](value: T):
        writer.write(value)

        @parameter
        if i < len(VariadicList(Ts)) - 1:
            writer.write(sep)

    values.each_idx[print_with_separator]()

    writer.write(end)

    # TODO: What is a flush function that works on CUDA?
    @parameter
    if not triple_is_nvidia_cuda():
        if flush:
            _flush(file=file)
