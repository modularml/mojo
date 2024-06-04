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
    os_is_windows,
    triple_is_nvidia_cuda,
    external_call,
    stdout,
)

from builtin.dtype import _get_dtype_printf_format
from builtin.builtin_list import _LITRefPackHelper
from builtin.file_descriptor import FileDescriptor
from memory import UnsafePointer

from utils import StringRef, unroll
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
                _dup(stream_id.value), mode.unsafe_ptr()
            )
        else:
            handle = external_call["fdopen", UnsafePointer[NoneType]](
                _dup(stream_id.value), mode.unsafe_ptr()
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
        _ = external_call["fflush", Int32](fd)


# ===----------------------------------------------------------------------=== #
#  _printf
# ===----------------------------------------------------------------------=== #


@no_inline
fn _printf[
    fmt: StringLiteral, *types: AnyType
](*arguments: *types, file: FileDescriptor = stdout):
    # The argument pack will contain references for each value in the pack,
    # but we want to pass their values directly into the C snprintf call. Load
    # all the members of the pack.
    var kgen_pack = _LITRefPackHelper(arguments._value).get_as_kgen_pack()

    # FIXME(37129): Cannot use get_loaded_kgen_pack because vtables on types
    # aren't stripped off correctly.
    var loaded_pack = __mlir_op.`kgen.pack.load`(kgen_pack)

    @parameter
    if triple_is_nvidia_cuda():
        _ = external_call["vprintf", Int32](
            fmt.unsafe_ptr(), UnsafePointer.address_of(loaded_pack)
        )
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
            ](fd, fmt.unsafe_ptr(), loaded_pack)


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
    var kgen_pack = _LITRefPackHelper(arguments._value).get_as_kgen_pack()

    # FIXME(37129): Cannot use get_loaded_kgen_pack because vtables on types
    # aren't stripped off correctly.
    var loaded_pack = __mlir_op.`kgen.pack.load`(kgen_pack)

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
        ](str, size, fmt.unsafe_ptr(), loaded_pack)
    )


@no_inline
fn _snprintf_scalar[
    type: DType,
    float_format: StringLiteral = "%.17g",
](buffer: UnsafePointer[UInt8], size: Int, x: Scalar[type]) -> Int:
    @parameter
    if type == DType.bool:
        if x:
            return _snprintf["True"](buffer, size)
        else:
            return _snprintf["False"](buffer, size)
    elif type.is_integral() or type == DType.address:
        return _snprintf[_get_dtype_printf_format[type]()](buffer, size, x)
    elif (
        type == DType.float16 or type == DType.bfloat16 or type == DType.float32
    ):
        # We need to cast the value to float64 to print it.
        return _float_repr[float_format](buffer, size, x.cast[DType.float64]())
    elif type == DType.float64:
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
    if type == DType.bool:
        _put["True"]() if x else _put["False"]()
    elif type.is_integral() or type == DType.address:
        _printf[format](x)
    elif type.is_floating_point():

        @parameter
        if triple_is_nvidia_cuda():
            _printf[format](x.cast[DType.float64]())
        else:
            _put(str(x))
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
        _put(str(x))


@no_inline
fn _put(x: String, file: FileDescriptor = stdout):
    # 'x' is borrowed, so we know it will outlive the call to print.
    _put(x._strref_dangerous(), file=file)


@no_inline
fn _put(x: StringRef, file: FileDescriptor = stdout):
    # Avoid printing "(null)" for an empty/default constructed `String`
    var str_len = len(x)

    if not str_len:
        return

    @parameter
    if triple_is_nvidia_cuda():
        var tmp = 0
        var arg_ptr = UnsafePointer.address_of(tmp)
        _ = external_call["vprintf", Int32](
            x.data, arg_ptr.bitcast[UnsafePointer[NoneType]]()
        )
    else:
        alias MAX_STR_LEN = 0x1000_0000

        # The string can be printed, so that's fine.
        if str_len < MAX_STR_LEN:
            _printf["%.*s"](x.length, x.data, file=file)
            return

        # The string is large, then we need to chunk it.
        var p = x.data
        while str_len:
            var ll = min(str_len, MAX_STR_LEN)
            _printf["%.*s"](ll, p, file=file)
            str_len -= ll
            p += ll


@no_inline
fn _put[x: StringLiteral](file: FileDescriptor = stdout):
    _put(StringRef(x), file=file)


@no_inline
fn _put(x: DType, file: FileDescriptor = stdout):
    _put(str(x), file=file)


# ===----------------------------------------------------------------------=== #
#  print
# ===----------------------------------------------------------------------=== #


@no_inline
fn print[
    *Ts: Stringable
](*values: *Ts, flush: Bool = False, file: FileDescriptor = stdout):
    """Prints elements to the text stream. Each element is separated by a
    whitespace and followed by a newline character.

    Parameters:
        Ts: The elements types.

    Args:
        values: The elements to print.
        flush: If set to true, then the stream is forcibly flushed.
        file: The output stream.
    """
    _print(values, sep=" ", end="\n", flush=flush, file=file)


@no_inline
fn print[
    *Ts: Stringable, EndTy: Stringable
](
    *values: *Ts,
    end: EndTy,
    flush: Bool = False,
    file: FileDescriptor = stdout,
):
    """Prints elements to the text stream. Each element is separated by a
    whitespace and followed by `end`.

    Parameters:
        Ts: The elements types.
        EndTy: The type of end argument.

    Args:
        values: The elements to print.
        end: The String to write after printing the elements.
        flush: If set to true, then the stream is forcibly flushed.
        file: The output stream.
    """
    _print(values, sep=" ", end=str(end), flush=flush, file=file)


@no_inline
fn print[
    SepTy: Stringable, *Ts: Stringable
](*values: *Ts, sep: SepTy, flush: Bool = False, file: FileDescriptor = stdout):
    """Prints elements to the text stream. Each element is separated by `sep`
    and followed by a newline character.

    Parameters:
        SepTy: The type of separator.
        Ts: The elements types.

    Args:
        values: The elements to print.
        sep: The separator used between elements.
        flush: If set to true, then the stream is forcibly flushed.
        file: The output stream.
    """
    _print(values, sep=str(sep), end="\n", flush=flush, file=file)


@no_inline
fn print[
    SepTy: Stringable, EndTy: Stringable, *Ts: Stringable
](
    *values: *Ts,
    sep: SepTy,
    end: EndTy,
    flush: Bool = False,
    file: FileDescriptor = stdout,
):
    """Prints elements to the text stream. Each element is separated by `sep`
    and followed by `end`.

    Parameters:
        SepTy: The type of separator.
        EndTy: The type of end argument.
        Ts: The elements types.

    Args:
        values: The elements to print.
        sep: The separator used between elements.
        end: The String to write after printing the elements.
        flush: If set to true, then the stream is forcibly flushed.
        file: The output stream.
    """
    _print(values, sep=str(sep), end=str(end), flush=flush, file=file)


@no_inline
fn _print[
    *Ts: Stringable
](
    values: VariadicPack[_, _, Stringable, Ts],
    *,
    sep: String,
    end: String,
    flush: Bool,
    file: FileDescriptor,
):
    @parameter
    fn print_with_separator[i: Int, T: Stringable](value: T):
        _put(str(value), file=file)

        @parameter
        if i < values.__len__() - 1:
            _put(sep, file=file)

    values.each_idx[print_with_separator]()

    _put(end, file=file)
    if flush:
        _flush(file=file)


# ===----------------------------------------------------------------------=== #
#  print_fmt
# ===----------------------------------------------------------------------=== #


# TODO:
#   Finish transition to using non-allocating formatting abstractions by
#   default, replace `print` with this function.
@no_inline
fn _print_fmt[
    T: Formattable,
    *Ts: Formattable,
    sep: StringLiteral = " ",
    end: StringLiteral = "\n",
](first: T, *rest: *Ts, flush: Bool = False):
    """Prints elements to the text stream. Each element is separated by `sep`
    and followed by `end`.

    This print function does not perform unnecessary intermediate String
    allocations during formatting.

    Parameters:
        T: The first element type.
        Ts: The remaining element types.
        sep: The separator used between elements.
        end: The String to write after printing the elements.

    Args:
        first: The first element.
        rest: The remaining elements.
        flush: If set to true, then the stream is forcibly flushed.
    """
    var writer = Formatter.stdout()

    write_to(writer, first)

    @parameter
    fn print_elt[T: Formattable](a: T):
        write_to(writer, sep, a)

    rest.each[print_elt]()

    write_to(writer, end)

    # TODO: What is a flush function that works on CUDA?
    @parameter
    if not triple_is_nvidia_cuda():
        if flush:
            _flush()
