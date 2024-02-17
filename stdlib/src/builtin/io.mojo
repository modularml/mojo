# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides utilities for working with input/output.

These are Mojo built-ins, so you don't need to import them.
"""

from os.atomic import Atomic
from sys import external_call
from sys.info import bitwidthof, os_is_windows, triple_is_nvidia_cuda

from algorithm.functional import unroll
from complex import ComplexSIMD as _ComplexSIMD
from math.math import align_up
from memory.unsafe import Pointer
from python.object import PythonObject
from tensor.tensor import Tensor

from utils.index import StaticIntTuple
from utils.list import DimList

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

    fn __init__(stream_id: Int) -> Self:
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
        return Self {handle: handle}

    fn __enter__(self) -> Self:
        return self

    fn __exit__(self):
        """Closes the file handle."""
        _ = external_call["fclose", Int32](self.handle)


# ===----------------------------------------------------------------------=== #
#  _printf
# ===----------------------------------------------------------------------=== #


@no_inline
fn _printf[*types: AnyRegType](fmt: StringLiteral, *arguments: *types):
    @parameter
    if triple_is_nvidia_cuda():
        # We need to make sure that the call to vprintf consistently uses
        # the same type, otherwise you end up with signature conflicts when
        # using external_call.
        var args = VariadicList(arguments)
        var args_ptr = Pointer.address_of(args)
        _ = external_call["vprintf", Int32](
            fmt.data(), args_ptr.bitcast[Pointer[Int]]().load()
        )
    else:
        with _fdopen(_fdopen.STDOUT) as fd:
            var num_characters_written = __mlir_op.`pop.external_call`[
                func = "KGEN_CompilerRT_fprintf".value,
                variadicType = __mlir_attr[
                    `(`,
                    `!kgen.pointer<none>,`,
                    `!kgen.pointer<scalar<si8>>`,
                    `) -> !pop.scalar<si32>`,
                ],
                _type=Int32,
            ](fd.handle.address, fmt.data(), arguments)
            # Note: currently ignoring errors if `fprintf` in the case that
            # fprintf returns a negative value.
            if num_characters_written > 0:
                _ = external_call["fflush", Int32](fd)


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
    # TODO(7585): Use `ord("-")` when it is possible at compile-time.
    alias minus: Int8 = 45  # '-'
    alias dot: Int8 = 46  # '.'
    if p.load() == minus:
        p += 1
    while p.load() != 0 and isdigit(p.load()):
        p += 1
    if p.load():
        return n
    p.store(dot)
    p += 1
    p.store(48)  # '0'
    p += 1
    p.store(0)
    return n + 2


@always_inline
fn _index_printf_format() -> StringLiteral:
    @parameter
    if bitwidthof[Int]() == 32:
        return "%d"
    elif os_is_windows():
        return "%lld"
    else:
        return "%ld"


@always_inline
fn _get_dtype_printf_format[type: DType]() -> StringLiteral:
    @parameter
    if type == DType.bool:
        return _index_printf_format()
    elif type == DType.uint8:
        return "%hhu"
    elif type == DType.int8:
        return "%hhi"
    elif type == DType.uint16:
        return "%hu"
    elif type == DType.int16:
        return "%hi"
    elif type == DType.uint32:
        return "%u"
    elif type == DType.int32:
        return "%i"
    elif type == DType.int64:

        @parameter
        if os_is_windows():
            return "%lld"
        else:
            return "%ld"
    elif type == DType.uint64:

        @parameter
        if os_is_windows():
            return "%llu"
        else:
            return "%lu"
    elif type == DType.index:
        return _index_printf_format()

    elif type == DType.address:
        return "%zx"

    elif type.is_floating_point():
        return "%.17g"

    else:
        constrained[False, "invalid dtype"]()

    return ""


@no_inline
fn _snprintf_int(
    buffer: Pointer[Int8],
    size: Int,
    x: Int,
) -> Int:
    return _snprintf(buffer, size, _index_printf_format(), x.value)


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
    elif type.is_integral():
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
#  _put
# ===----------------------------------------------------------------------=== #


@no_inline
fn _put(x: Int):
    """Prints a scalar value.

    Args:
        x: The value to print.
    """
    _printf(_index_printf_format(), x)


@no_inline
fn _put_simd_scalar[type: DType](x: SIMD[type, 1]):
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
    elif type.is_integral():
        _printf(format, x)
    elif type.is_floating_point():

        @parameter
        if triple_is_nvidia_cuda():
            _printf(format, x.cast[DType.float64]())
        else:
            _put(String(x))
    elif type == DType.address:
        _printf(format, x)
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
    if len(x) > 0:
        _printf("%.*s", x.length, x.data)


@no_inline
fn _put(x: StringLiteral):
    _put(StringRef(x))


@no_inline
fn _put(x: DType):
    _put(x.__str__())


@no_inline
fn put_new_line():
    """Prints a new line character."""
    _printf("\n")


# ===----------------------------------------------------------------------=== #
#  print
# ===----------------------------------------------------------------------=== #


@no_inline
fn print():
    """Prints a newline."""
    put_new_line()


# These specific overloads are defined for twofold:
# 1. Reduce binary size
# 2. The `Stringable` path in variadic print doesn't work on GPUs.


@no_inline
fn print(t: DType):
    """Prints a DType.

    Args:
        t: The DType to print.
    """
    print(t.__str__())


@no_inline
fn print(x: String):
    """Prints a string.

    Args:
        x: The string to print.
    """
    _put(x)
    put_new_line()


@no_inline
fn print(x: StringRef):
    """Prints a string.

    Args:
        x: The string to print.
    """
    _put(x)
    put_new_line()


@no_inline
fn print(x: StringLiteral):
    """Prints a string.

    Args:
        x: The string to print.
    """
    _put(StringRef(x))
    put_new_line()


@no_inline
fn print(x: Bool):
    """Prints a boolean value.

    Args:
        x: The value to print.
    """
    _put("True") if x else _put("False")
    put_new_line()


@no_inline
fn print(x: FloatLiteral):
    """Prints a float literal.

    Args:
        x: The value to print.
    """
    print(Float64(x))


@no_inline
fn print(x: Int):
    """Prints an integer value.

    Args:
        x: The value to print.
    """
    _put(x)
    put_new_line()


@no_inline
fn print[
    simd_width: Int,
    type: DType,
](vec: SIMD[type, simd_width]):
    """Prints a SIMD value.

    Parameters:
        simd_width: The SIMD vector width.
        type: The DType of the value.

    Args:
        vec: The SIMD value to print.
    """

    _put(vec)
    put_new_line()


@no_inline
fn print[
    simd_width: Int,
    type: DType,
](vec: _ComplexSIMD[type, simd_width]):
    """Prints a SIMD value.

    Parameters:
        simd_width: The SIMD vector width.
        type: The DType of the value.

    Args:
        vec: The complex value to print.
    """
    print(String(vec))


@no_inline
fn print[type: DType](x: Atomic[type]):
    """Prints an atomic value.

    Parameters:
        type: The DType of the atomic value.

    Args:
        x: The value to print.
    """
    _put(x.value)
    put_new_line()


@no_inline
fn print[length: Int](shape: DimList):
    """Prints a DimList object.

    Parameters:
        length: The length of the DimList.

    Args:
        shape: The DimList object to print.
    """

    @always_inline
    @parameter
    fn _print_elem[idx: Int]():
        var value = shape.at[idx]()

        @parameter
        if idx != 0:
            _printf(", ")
        _put(value.get().value)

    _put("[")
    unroll[_print_elem, length]()
    _put("]")
    put_new_line()


@no_inline
fn print(obj: object):
    """Prints an object type.

    Args:
        obj: The object to print.
    """
    obj.print()
    put_new_line()


@no_inline
fn print(err: Error):
    """Prints an Error type.

    Args:
        err: The Error to print.
    """
    print(err.__str__())


# ===----------------------------------------------------------------------=== #
#  variadic print
# ===----------------------------------------------------------------------=== #


struct _StringableTuple[*Ts: Stringable](Sized):
    alias _type = __mlir_type[
        `!kgen.pack<:variadic<`, Stringable, `> `, Ts, `>`
    ]
    var storage: Self._type

    fn __init__(inout self, value: Self._type):
        self.storage = value

    @staticmethod
    fn _offset[i: Int]() -> Int:
        constrained[i >= 0, "index must be positive"]()

        @parameter
        if i == 0:
            return 0
        else:
            return align_up(
                Self._offset[i - 1]()
                + align_up(sizeof[Ts[i - 1]](), alignof[Ts[i - 1]]()),
                alignof[Ts[i]](),
            )

    fn _print[i: Int](inout self):
        _put(" ")
        _put(self._at[i]())

    fn _at[i: Int](inout self) -> String:
        alias offset = Self._offset[i]()
        var addr = Pointer.address_of(self).bitcast[Int8]().offset(offset)
        var ptr = __mlir_op.`pop.pointer.bitcast`[
            _type = __mlir_type[`!kgen.pointer<:`, Stringable, ` `, Ts[i], `>`]
        ](addr.address)

        return str(__get_address_as_lvalue(ptr))

    fn __len__(self) -> Int:
        return len(VariadicList(Ts))


fn _print_elements[
    T: Stringable, *Ts: Stringable
](first: T, inout rest: _StringableTuple[Ts]):
    _put(first.__str__())

    @parameter
    fn each[i: Int]():
        rest._print[i]()

    unroll[each, len(VariadicList(Ts))]()


@no_inline
fn print[T: Stringable, *Ts: Stringable](first: T, *rest: *Ts):
    """Prints a sequence of elements, joined by spaces, followed by a newline.

    Parameters:
        T: The first element type.
        Ts: The remaining element types.

    Args:
        first: The first element.
        rest: The remaining elements.
    """
    var vals = _StringableTuple[Ts](rest)
    _print_elements(first, vals)
    put_new_line()


# FIXME(#8843, #12811): This should be removed, and instead implemented in terms
# of `print` and keyword arguments.
@no_inline
fn print_no_newline[T: Stringable, *Ts: Stringable](first: T, *rest: *Ts):
    """Prints a sequence of elements, joined by spaces.

    Parameters:
        T: The first element type.
        Ts: The remaining element types.

    Args:
        first: The first element.
        rest: The remaining elements.
    """
    var vals = _StringableTuple[Ts](rest)
    _print_elements(first, vals)
