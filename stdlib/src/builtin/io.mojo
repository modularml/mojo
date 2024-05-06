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

from sys import os_is_windows, triple_is_nvidia_cuda, external_call

from builtin.dtype import _get_dtype_printf_format
from builtin.builtin_list import _LITRefPackHelper
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

    fn __init__(inout self, stream_id: Int):
        """Creates a file handle to the stdout/stderr stream.

        Args:
            stream_id: The stream id (either `STDOUT` or `STDERR`)
        """
        alias mode = "a"
        var handle: UnsafePointer[NoneType]

        @parameter
        if os_is_windows():
            handle = external_call["_fdopen", UnsafePointer[NoneType]](
                _dup(stream_id), mode.unsafe_ptr()
            )
        else:
            handle = external_call["fdopen", UnsafePointer[NoneType]](
                _dup(stream_id), mode.unsafe_ptr()
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
fn _flush():
    with _fdopen(_fdopen.STDOUT) as fd:
        _ = external_call["fflush", Int32](fd)


# ===----------------------------------------------------------------------=== #
#  _printf
# ===----------------------------------------------------------------------=== #


@no_inline
fn _printf[*types: AnyType](fmt: StringLiteral, *arguments: *types):
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
        with _fdopen(_fdopen.STDOUT) as fd:
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
    *types: AnyType
](
    str: UnsafePointer[Int8],
    size: Int,
    fmt: StringLiteral,
    *arguments: *types,
) -> Int:
    """Writes a format string into an output pointer.

    Args:
        str: A pointer into which the format string is written.
        size: At most, `size - 1` bytes are written into the output string.
        fmt: A format string.
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
    type: DType
](buffer: UnsafePointer[Int8], size: Int, x: Scalar[type],) -> Int:
    alias format = _get_dtype_printf_format[type]()

    @parameter
    if type == DType.bool:
        if x:
            return _snprintf(buffer, size, "True")
        else:
            return _snprintf(buffer, size, "False")
    elif type.is_integral() or type == DType.address:
        return _snprintf(buffer, size, format, x)
    elif (
        type == DType.float16 or type == DType.bfloat16 or type == DType.float32
    ):
        # We need to cast the value to float64 to print it.
        return _float_repr(buffer, size, x.cast[DType.float64]())
    elif type == DType.float64:
        return _float_repr(buffer, size, rebind[Float64](x))
    return 0


# ===----------------------------------------------------------------------=== #
#  Helper functions to print a single pop scalar without spacing or new line.
# ===----------------------------------------------------------------------=== #


@no_inline
fn _float_repr(buffer: UnsafePointer[Int8], size: Int, x: Float64) -> Int:
    # Using `%.17g` with decimal check is equivalent to CPython's fallback path
    # when its more complex dtoa library (forked from
    # https://github.com/dtolnay/dtoa) is not available.
    var n = _snprintf(buffer, size, "%.17g", x.value)
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
fn _put(x: Int):
    """Prints a scalar value.

    Args:
        x: The value to print.
    """
    _printf(_get_dtype_printf_format[DType.index](), x)


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
        _put("True") if x else _put("False")
    elif type.is_integral() or type == DType.address:
        _printf(format, x)
    elif type.is_floating_point():

        @parameter
        if triple_is_nvidia_cuda():
            _printf(format, x.cast[DType.float64]())
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
        _put("[")

        @unroll
        for i in range(simd_width):
            _put_simd_scalar(x[i])
            if i != simd_width - 1:
                _put(", ")
        _put("]")
    else:
        _put(String(x))


@no_inline
fn _put(x: String):
    # 'x' is borrowed, so we know it will outlive the call to print.
    _put(x._strref_dangerous())


@no_inline
fn _put(x: StringRef):
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
            _printf("%.*s", x.length, x.data)
            return

        # The string is large, then we need to chunk it.
        var p = x.data
        while str_len:
            var ll = min(str_len, MAX_STR_LEN)
            _printf("%.*s", ll, p)
            str_len -= ll
            p += ll


@no_inline
fn _put(x: StringLiteral):
    _put(StringRef(x))


@no_inline
fn _put(x: DType):
    _put(str(x))


# ===----------------------------------------------------------------------=== #
#  print
# ===----------------------------------------------------------------------=== #


@no_inline
fn print[
    *Ts: Stringable
](
    *values: *Ts,
    sep: StringLiteral = " ",
    end: StringLiteral = "\n",
    flush: Bool = False,
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
    """

    @parameter
    fn print_with_separator[i: Int, T: Stringable](value: T):
        _put(value)

        @parameter
        if i < values.__len__() - 1:
            _put(sep)

    values.each_idx[print_with_separator]()

    _put(end)
    if flush:
        _flush()


# ===----------------------------------------------------------------------=== #
#  print_fmt
# ===----------------------------------------------------------------------=== #


# TODO:
#   Finish transition to using non-allocating formatting abstractions by
#   default, replace `print` with this function.
@no_inline
fn _print_fmt[
    T: Formattable, *Ts: Formattable
](
    first: T,
    *rest: *Ts,
    sep: StringLiteral = " ",
    end: StringLiteral = "\n",
    flush: Bool = False,
):
    """Prints elements to the text stream. Each element is separated by `sep`
    and followed by `end`.

    This print function does not perform unnecessary intermediate String
    allocations during formatting.

    Parameters:
        T: The first element type.
        Ts: The remaining element types.

    Args:
        first: The first element.
        rest: The remaining elements.
        sep: The separator used between elements.
        end: The String to write after printing the elements.
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
