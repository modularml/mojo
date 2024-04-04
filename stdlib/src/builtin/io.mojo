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

from sys import external_call
from sys.info import bitwidthof, os_is_windows, triple_is_nvidia_cuda

from builtin.dtype import _get_dtype_printf_format
from memory.unsafe import Pointer

from utils import StringRef, unroll

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
fn _align_up(value: Int, alignment: Int) -> Int:
    var div_ceil = (value + alignment - 1)._positive_div(alignment)
    return div_ceil * alignment


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
    var handle: Pointer[NoneType]

    fn __init__(inout self, stream_id: Int):
        """Creates a file handle to the stdout/stderr stream.

        Args:
            stream_id: The stream id (either `STDOUT` or `STDERR`)
        """
        alias mode = "a"
        var handle: Pointer[NoneType]

        @parameter
        if os_is_windows():
            handle = external_call["_fdopen", Pointer[NoneType]](
                _dup(stream_id), mode.data()
            )
        else:
            handle = external_call["fdopen", Pointer[NoneType]](
                _dup(stream_id), mode.data()
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
fn _printf[*types: AnyRegType](fmt: StringLiteral, *arguments: *types):
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
        ](fd, fmt.data(), arguments)


# ===----------------------------------------------------------------------=== #
#  _snprintf
# ===----------------------------------------------------------------------=== #


@no_inline
fn _snprintf[
    *types: AnyRegType
](str: Pointer[Int8], size: Int, fmt: StringLiteral, *arguments: *types) -> Int:
    """Writes a format string into an output pointer.

    Args:
        str: A pointer into which the format string is written.
        size: At most, `size - 1` bytes are written into the output string.
        fmt: A format string.
        arguments: Arguments interpolated into the format string.

    Returns:
        The number of bytes written into the output string.
    """
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
        ](str, size, fmt.data(), arguments)
    )


@no_inline
fn _snprintf_int(
    buffer: Pointer[Int8],
    size: Int,
    x: Int,
) -> Int:
    return _snprintf(
        buffer, size, _get_dtype_printf_format[DType.index](), x.value
    )


@no_inline
fn _snprintf_scalar[
    type: DType
](buffer: Pointer[Int8], size: Int, x: Scalar[type],) -> Int:
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
fn _float_repr(buffer: Pointer[Int8], size: Int, x: Float64) -> Int:
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
    if p.load() == minus:
        p += 1
    while p.load() != 0 and isdigit(p.load()):
        p += 1
    if p.load():
        return n
    p.store(dot)
    p += 1
    p.store(ord("0"))
    p += 1
    p.store(0)
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


fn _min(x: Int, y: Int) -> Int:
    return x if x < y else y


@no_inline
fn _put(x: StringRef):
    # Avoid printing "(null)" for an empty/default constructed `String`
    var str_len = len(x)

    if not str_len:
        return

    alias MAX_STR_LEN = 0x1000_0000

    # The string can be printed, so that's fine.
    if str_len < MAX_STR_LEN:
        _printf("%.*s", x.length, x.data)
        return

    # The string is large, then we need to chunk it.
    var p = x.data
    while str_len:
        var ll = _min(str_len, MAX_STR_LEN)
        _printf("%.*s", ll, p)
        str_len -= ll
        p += ll


@no_inline
fn _put(x: StringLiteral):
    _put(StringRef(x))


@no_inline
fn _put(x: DType):
    _put(x.__str__())


# ===----------------------------------------------------------------------=== #
#  print
# ===----------------------------------------------------------------------=== #


@no_inline
fn print(
    *, sep: StringLiteral = " ", end: StringLiteral = "\n", flush: Bool = False
):
    """Prints the end value.

    Args:
        sep: The separator used between elements.
        end: The String to write after printing the elements.
        flush: If set to true, then the stream is forcibly flushed.
    """
    _put(end)
    if flush:
        _flush()


@no_inline
fn print[
    T: Stringable, *Ts: Stringable
](
    first: T,
    *rest: *Ts,
    sep: StringLiteral = " ",
    end: StringLiteral = "\n",
    flush: Bool = False,
):
    """Prints elements to the text stream. Each element is separated by `sep`
    and followed by `end`.

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
    _put(str(first))

    @parameter
    fn print_elt[T: Stringable](a: T):
        _put(sep)
        _put(a)

    rest.each[print_elt]()

    _put(end)
    if flush:
        _flush()
