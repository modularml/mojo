# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Implements the Error class.

These are Mojo built-ins, so you don't need to import them.
"""

from std.format._utils import FormatStruct
from std.memory import (
    ArcPointer,
    OwnedPointer,
)
from std.ffi import CStringSlice, external_call
import std.format._utils as fmt
from std.sys import is_gpu
from std.sys.info import size_of, align_of


# ===-----------------------------------------------------------------------===#
# StackTrace
# ===-----------------------------------------------------------------------===#


struct StackTrace(ImplicitlyCopyable, Movable, Writable):
    """Holds a stack trace captured at a specific location.

    A `StackTrace` instance always contains a valid stack trace. Use the
    `collect_if_enabled()` static method to conditionally capture a stack
    trace, which returns `None` if stack trace collection is disabled or
    unavailable.

    Copying a `StackTrace` shares the captured trace instead of duplicating it,
    so copies cost a reference count increment.
    """

    var _data: ArcPointer[OwnedPointer[UInt8]]
    """A reference-counted, owning pointer to a null-terminated C string
    containing the stack trace."""

    def __init__(
        out self,
        *,
        unsafe_from_raw_pointer: Pointer[UInt8, MutUntrackedOrigin],
    ):
        """Construct a StackTrace from a raw pointer to a C string.

        Args:
            unsafe_from_raw_pointer: A pointer to a null-terminated C string
                containing the stack trace. The StackTrace takes ownership.

        Safety:
            The pointer must be valid and point to a null-terminated string.
            The caller transfers ownership to this StackTrace.
        """
        self._data = ArcPointer(
            OwnedPointer(unsafe_from_raw_pointer=unsafe_from_raw_pointer)
        )

    @staticmethod
    @no_inline
    def collect_if_enabled(depth: Int = 0) -> Optional[StackTrace]:
        """Collect a stack trace if enabled by configuration.

        This method checks the `max-debug.stack-trace-on-error` Config key
        (settable via the `MODULAR_DEBUG=stack-trace-on-error` environment
        variable) and collects a stack trace only if it is enabled. Returns
        `None` if stack traces are disabled, on GPU, or if collection fails.

        Args:
            depth: The maximum depth of the stack trace to collect.
                   When `depth` is zero, the entire stack trace is collected.
                   When `depth` is negative, no stack trace is collected.

        Returns:
            An `Optional[StackTrace]` containing the stack trace if collection
            succeeded, or `None` if disabled or unavailable.
        """

        comptime if is_gpu():
            return None

        if depth < 0:
            return None

        var buffer = Optional[Pointer[UInt8, MutUntrackedOrigin]]()
        var num_bytes = external_call[
            "KGEN_CompilerRT_GetStackTrace", SIMDLength
        ](Pointer(to=buffer), depth)
        # When num_bytes is zero, the stack trace was not collected.
        if num_bytes == 0:
            return None

        # If num_bytes > 0, `buffer` will not be left null.
        return StackTrace(unsafe_from_raw_pointer=buffer.unsafe_value())

    def write_to(self, mut writer: Some[Writer]):
        """Writes the StackTrace to the provided Writer.

        Args:
            writer: The object to write to.
        """
        writer.write_string(
            StringSlice(
                unsafe_from_utf8=CStringSlice(
                    unsafe_from_ptr=self._data[].ptr().unsafe_bitcast[Int8]()
                )
            )
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the StackTrace to the provided Writer in repr format.

        Args:
            writer: The object to write to.
        """
        FormatStruct(writer, "StackTrace").fields(self)


# ===-----------------------------------------------------------------------===#
# Error
# ===-----------------------------------------------------------------------===#


struct Error(
    ImplicitlyCopyable,
    Writable,
):
    """This type represents an Error.

    An `Error` is cheap to copy: the message and the optional stack trace are
    both reference counted, so re-raising a caught error with `raise e` costs a
    reference count increment. Transfer with `raise e^` to avoid even that.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _error: String
    """The backing error message."""

    var _stack_trace: Optional[StackTrace]
    """The stack trace of the error, if collected.

    By default, stack trace is collected for errors created from string
    literals. Stack trace collection can be controlled via the
    `max-debug.stack-trace-on-error` Config key (settable via the
    `MODULAR_DEBUG=stack-trace-on-error` environment variable).
    """

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    @implicit
    def __init__(out self, var value: String, *, depth: Int = -1):
        """Construct an Error object with a given String.

        Args:
            value: The error message.
            depth: The depth of the stack trace to collect. When negative,
                no stack trace is collected.
        """
        self._error = value^
        self._stack_trace = StackTrace.collect_if_enabled(depth)

    @always_inline
    @implicit
    def __init__(out self, value: StringLiteral):
        """Construct an Error object with a given string literal.

        Args:
            value: The error message.
        """
        self._error = String(value)
        self._stack_trace = StackTrace.collect_if_enabled(0)

    @implicit
    @no_inline
    def __init__[*Ts: Writable](out self, *args: *Ts):
        """Construct an Error by concatenating a sequence of Writable arguments.

        Args:
            args: A sequence of Writable arguments.

        Parameters:
            Ts: The types of the arguments to format. Each type must be satisfy
                `Writable`.
        """
        self = Error(String(*args), depth=0)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """
        Formats this error to the provided Writer.

        Args:
            writer: The object to write to.
        """
        self._error.write_to(writer)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """
        Formats this error to the provided Writer.

        Args:
            writer: The object to write to.
        """
        fmt.FormatStruct(writer, "Error").fields(fmt.Repr(self._error))

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def get_stack_trace(self) -> Optional[String]:
        """Returns the stack trace of the error, if available.

        Returns:
            An `Optional[String]` containing the stack trace if one was
            collected, or `None` if stack trace collection was disabled
            or unavailable.
        """
        if self._stack_trace:
            return String(self._stack_trace.value())
        return None


@doc_hidden
def __mojo_debugger_raise_hook():
    """This function is used internally by the Mojo Debugger."""
    pass
