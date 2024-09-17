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
"""Implements a formatter abstraction for objects that can format
themselves to a string.
"""

from builtin.io import _put
from memory import UnsafePointer

# ===----------------------------------------------------------------------===#
# Interface traits
# ===----------------------------------------------------------------------===#


trait Formattable:
    """
    The `Formattable` trait describes a type that can be converted to a stream
    of UTF-8 encoded data by writing to a formatter object.

    Examples:

    Implement `Formattable` and `Stringable` for a type:

    ```mojo
    struct Point(Stringable, Formattable):
        var x: Float64
        var y: Float64

        fn __str__(self) -> String:
            return String.format_sequence(self)

        fn format_to(self, inout writer: Formatter):
            writer.write("(", self.x, ", ", self.y, ")")
    ```
    """

    fn format_to(self, inout writer: Formatter):
        """
        Formats the string representation of this type to the provided formatter.

        Args:
            writer: The formatter to write to.
        """
        ...


trait ToFormatter:
    """
    The `ToFormatter` trait describes a type that can be written to by a
    `Formatter` object.
    """

    fn _unsafe_to_formatter(inout self) -> Formatter:
        ...


# ===----------------------------------------------------------------------===#
# Formatter
# ===----------------------------------------------------------------------===#


struct Formatter:
    """
    A `Formatter` is used by types implementing the `Formattable` trait to write
    bytes to the underlying formatter output buffer or stream.
    """

    # FIXME(#37996):
    #   This manual implementation of a closure function ptr + closure data
    #   arg is needed to workaround a bug with `escaping` closure capture values
    #   seemingly getting clobbered in between when the closure was constructed
    #   and first called. Once that bug is fixed, this should be replaced with
    #   an `escaping` closure again.
    var _write_func: fn (UnsafePointer[NoneType], StringRef) -> None
    var _write_func_arg: UnsafePointer[NoneType]
    """Closure argument passed to `_write_func`."""

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    fn __init__[F: ToFormatter](inout self, inout output: F):
        """Construct a new `Formatter` from a value implementing `ToFormatter`.

        Parameters:
            F: The type that supports being used to back a `Formatter`.

        Args:
            output: Value to accumulate or process output streamed to the `Formatter`.
        """
        self = output._unsafe_to_formatter()

    fn __init__(inout self, *, fd: FileDescriptor):
        """
        Constructs a `Formatter` that writes to the given file descriptor.

        Args:
            fd: The file descriptor to write to.
        """

        @always_inline
        fn write_to_fd(ptr: UnsafePointer[NoneType], strref: StringRef):
            var fd0 = ptr.bitcast[FileDescriptor]()[].value

            _put(strref, file=fd0)

        self = Formatter(
            write_to_fd,
            UnsafePointer.address_of(fd).bitcast[NoneType](),
        )

    fn __init__(
        inout self,
        func: fn (UnsafePointer[NoneType], StringRef) -> None,
        arg: UnsafePointer[NoneType],
    ):
        """Constructs a formatter from any closure that accepts `StringRef`s.

        This function should only be used by low-level types that wish to
        accept streamed formatted data.

        Args:
            func: Raw closure function pointer.
            arg: Opaque user data argument that is passed to the closure function pointer.
        """
        self._write_func = func
        self._write_func_arg = arg

    fn __moveinit__(inout self, owned other: Self):
        """Move this value.

        Args:
            other: The value to move.
        """
        self._write_func = other._write_func
        self._write_func_arg = other._write_func_arg

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    # TODO: Constrain to only require an immutable StringSlice[..]`
    @always_inline
    fn write_str(inout self, str_slice: StringSlice[_]):
        """
        Write a string slice to this formatter.

        Args:
            str_slice: The string slice to write to this formatter. Must NOT be
              null terminated.
        """

        # SAFETY:
        #   Safe because `str_slice` is a `borrowed` arg, and so alive at least
        #   as long as this call.
        var strref: StringRef = str_slice._strref_dangerous()

        self._write_func(self._write_func_arg, strref)

    fn write[*Ts: Formattable](inout self: Formatter, *args: *Ts):
        """Write a sequence of formattable arguments to the provided formatter.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this formatter.
        """

        @parameter
        fn write_arg[T: Formattable](arg: T):
            arg.format_to(self)

        args.each[write_arg]()

    fn _write_int_padded(inout self, value: Int, *, width: Int):
        var int_width = value._decimal_digit_count()

        # TODO: Assumes user wants right-aligned content.
        if int_width < width:
            self._write_repeated(
                " ".as_string_slice(),
                width - int_width,
            )

        self.write(value)

    fn _write_repeated(inout self, str: StringSlice, count: Int):
        for _ in range(count):
            self.write_str(str)

    # ===------------------------------------------------------------------=== #
    # Factory methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    @staticmethod
    fn stdout() -> Self:
        """
        Constructs a formatter that writes directly to stdout.

        Returns:
            A formatter that writes provided data to the operating system
            standard output stream.
        """

        @always_inline
        fn write_to_stdout(_data: UnsafePointer[NoneType], strref: StringRef):
            _put(strref)

        return Formatter(write_to_stdout, UnsafePointer[NoneType]())
