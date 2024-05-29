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

# ===----------------------------------------------------------------------===#
# Interface traits
# ===----------------------------------------------------------------------===#


trait Formattable:
    """
    The `Formattable` trait describes a type that can be converted to a stream
    of UTF-8 encoded data by writing to a formatter object.
    """

    fn format_to(self, inout writer: Formatter):
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
        self = output._unsafe_to_formatter()

    fn __init__(
        inout self,
        func: fn (UnsafePointer[NoneType], StringRef) -> None,
        arg: UnsafePointer[NoneType],
    ):
        """
        Constructs a formatter from any closure that accepts string refs.
        """
        self._write_func = func
        self._write_func_arg = arg

    fn __moveinit__(inout self, owned other: Self):
        self._write_func = other._write_func
        self._write_func_arg = other._write_func_arg

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    # TODO(cleanup):
    #   Remove this overload by defining a working
    #   `StringSlice.__init__(StringLiteral)` implicit conversion.
    @always_inline
    fn write_str(inout self, literal: StringLiteral):
        """
        Write a string literal to this formatter.

        Args:
            literal: The string literal to write.
        """

        self.write_str(literal.as_string_slice())

    @always_inline
    fn write_str(inout self, str_slice: StringSlice[False, _]):
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

    # ===------------------------------------------------------------------=== #
    # Factory methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    @staticmethod
    fn stdout() -> Self:
        """
        Constructs a formatter that writes directly to stdout.
        """

        @always_inline
        fn write_to_stdout(_data: UnsafePointer[NoneType], strref: StringRef):
            _put(strref)

        return Formatter(write_to_stdout, UnsafePointer[NoneType]())


fn write_to[*Ts: Formattable](inout writer: Formatter, *args: *Ts):
    """
    Write a sequence of formattable arguments to the provided formatter.
    """

    @parameter
    fn write_arg[T: Formattable](arg: T):
        arg.format_to(writer)

    args.each[write_arg]()
