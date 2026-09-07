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

"""Provides logging functionality with different severity levels.

This module implements a simple logging system with configurable severity
levels: `NOTSET`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`. The
logging level can be set via the LOGGING_LEVEL environment variable.

The main components are:

- `Level`: An enum-like struct defining the available logging levels
- `Logger`: A struct that handles logging messages with different severity levels

Example:

```mojo
from std.logger import Logger

var logger = Logger()  # Uses default level from LOGGING_LEVEL env var
logger.info("Starting process")
logger.debug("Debug information")
logger.error("An error occurred")
```

The logger can be configured to write to different file descriptors (default
stdout). Messages below the configured level will be silently ignored.
"""

import std.sys
from std.format._utils import _FlushingWriteBuffer
from std.os import abort
from std.sys.defines import get_defined_string
from std.utils._ansi import Text, Color

from std.reflection import call_location, SourceLocation

# ===-----------------------------------------------------------------------===#
# DEFAULT_LEVEL
# ===-----------------------------------------------------------------------===#

comptime DEFAULT_LEVEL = Level._from_str(
    get_defined_string["LOGGING_LEVEL", "NOTSET"]()
)
"""The default logging level, determined by the LOGGING_LEVEL environment variable."""

# ===-----------------------------------------------------------------------===#
# Level
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Level(
    Comparable,
    ImplicitlyCopyable,
    Writable,
):
    """Represents logging severity levels.

    Defines the available logging levels in ascending order of severity.
    """

    var _value: Int

    comptime NOTSET = Self(0)
    """Lowest level, used when no level is set."""

    comptime TRACE = Self(10)
    """Repetitive trace information, Indicates repeated execution or IO-coupled
    activity, typically only of interest when diagnosing hangs or ensuring a
    section of code is executing."""

    comptime DEBUG = Self(20)
    """Detailed information, typically of interest only when diagnosing problems."""

    comptime INFO = Self(30)
    """Confirmation that things are working as expected."""

    comptime WARNING = Self(40)
    """Indication that something unexpected happened, or may happen in the near future."""

    comptime ERROR = Self(50)
    """Due to a more serious problem, the software has not been able to perform some function."""

    comptime CRITICAL = Self(60)
    """A serious error indicating that the program itself may be unable to continue running."""

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        """Returns True if this level equals the other level.

        Args:
            other: The level to compare with.

        Returns:
            Bool: True if the levels are equal, False otherwise.
        """
        return self._value == other._value

    def __lt__(self, other: Self) -> Bool:
        """Returns True if this level is less than the other level.

        Args:
            other: The level to compare with.

        Returns:
            Bool: True if this level is less than the other level, False otherwise.
        """
        return self._value < other._value

    def color(self) -> Color:
        """Returns the ANSI color of the level.

        Returns:
            The corresponding Color of the level.
        """
        if self == Self.TRACE:
            return Color.GREEN
        if self == Self.DEBUG:
            return Color.GREEN
        if self == Self.INFO:
            return Color.YELLOW
        if self == Self.WARNING:
            return Color.BLUE
        if self == Self.ERROR:
            return Color.MAGENTA
        if self == Self.CRITICAL:
            return Color.RED

        return Color("")

    @staticmethod
    def _from_str(name: StringSlice) -> Self:
        """Converts a string level name to a Level value.

        Args:
            name: The level name as a string (case insensitive).

        Returns:
            The corresponding Level value, or NOTSET if not recognized.
        """
        var lname = name.lower()
        if lname == "notset":
            return Self.NOTSET
        if lname == "trace":
            return Self.TRACE
        if lname == "debug":
            return Self.DEBUG
        if lname == "info":
            return Self.INFO
        if lname == "warning":
            return Self.WARNING
        if lname == "error":
            return Self.ERROR
        if lname == "critical":
            return Self.CRITICAL
        return Self.NOTSET

    def write_to(self, mut writer: Some[Writer]):
        """Writes the string representation of this level to a writer.

        Args:
            writer: The writer to write to.
        """
        if self == Self.NOTSET:
            writer.write("NOTSET")
        elif self == Self.TRACE:
            writer.write("TRACE")
        elif self == Self.DEBUG:
            writer.write("DEBUG")
        elif self == Self.INFO:
            writer.write("INFO")
        elif self == Self.WARNING:
            writer.write("WARNING")
        elif self == Self.ERROR:
            writer.write("ERROR")
        elif self == Self.CRITICAL:
            writer.write("CRITICAL")


# ===-----------------------------------------------------------------------===#
# Logger
# ===-----------------------------------------------------------------------===#


struct Logger[level: Level = DEFAULT_LEVEL](ImplicitlyCopyable):
    """A logger that outputs messages at or above a specified severity level.

    Parameters:
        level: The minimum severity level for messages to be logged.
    """

    var _fd: FileDescriptor
    var _prefix: String
    var _source_location: Bool

    def __init__(
        out self,
        fd: FileDescriptor = std.sys.stdout,
        *,
        prefix: String = "",
        source_location: Bool = False,
    ):
        """Initializes a new Logger.

        Args:
            fd: The file descriptor to write log messages to (defaults to stdout).
            prefix: The prefix to prepend to each log message (defaults to an
                empty string).
            source_location: Whether to include the source location in the log
                message (defaults to False).
        """
        self._fd = fd
        self._prefix = prefix
        self._source_location = source_location

    @always_inline
    @staticmethod
    def _is_disabled[target_level: Level]() -> Bool:
        """Returns True if logging at the target level is disabled.

        Parameters:
            target_level: The level to check if disabled.

        Returns:
            True if logging at the target level is disabled, False otherwise.
        """

        comptime if Self.level == Level.NOTSET:
            return True
        return Self.level > target_level

    @always_inline
    def trace[
        *Ts: Writable
    ](
        self,
        *values: *Ts,
        sep: StaticString = " ",
        end: StaticString = "\n",
        location: Optional[SourceLocation] = None,
    ):
        """Logs a trace message.

        Parameters:
            Ts: The types of values to log.

        Args:
            values: The values to log.
            sep: The separator to use between values (defaults to a space).
            end: The string to append to the end of the message
                (defaults to a newline).
            location: The location of the error (defaults to `call_location`).
        """
        comptime target_level = Level.TRACE

        comptime if not Self._is_disabled[target_level]():
            self._write_out[target_level](
                *values,
                sep=sep,
                end=end,
                location=location.or_else(call_location()),
            )

    @always_inline
    def debug[
        *Ts: Writable
    ](
        self,
        *values: *Ts,
        sep: StaticString = " ",
        end: StaticString = "\n",
        location: Optional[SourceLocation] = None,
    ):
        """Logs a debug message.

        Parameters:
            Ts: The types of values to log.

        Args:
            values: The values to log.
            sep: The separator to use between values (defaults to a space).
            end: The string to append to the end of the message
                (defaults to a newline).
            location: The location of the error (defaults to `call_location`).
        """
        comptime target_level = Level.DEBUG

        comptime if not Self._is_disabled[target_level]():
            self._write_out[target_level](
                *values,
                sep=sep,
                end=end,
                location=location.or_else(call_location()),
            )

    @always_inline
    def info[
        *Ts: Writable
    ](
        self,
        *values: *Ts,
        sep: StaticString = " ",
        end: StaticString = "\n",
        location: Optional[SourceLocation] = None,
    ):
        """Logs an info message.

        Parameters:
            Ts: The types of values to log.

        Args:
            values: The values to log.
            sep: The separator to use between values (defaults to a space).
            end: The string to append to the end of the message
                (defaults to a newline).
            location: The location of the error (defaults to `call_location`).
        """
        comptime target_level = Level.INFO

        comptime if not Self._is_disabled[target_level]():
            self._write_out[target_level](
                *values,
                sep=sep,
                end=end,
                location=location.or_else(call_location()),
            )

    @always_inline
    def warning[
        *Ts: Writable
    ](
        self,
        *values: *Ts,
        sep: StaticString = " ",
        end: StaticString = "\n",
        location: Optional[SourceLocation] = None,
    ):
        """Logs a warning message.

        Parameters:
            Ts: The types of values to log.

        Args:
            values: The values to log.
            sep: The separator to use between values (defaults to a space).
            end: The string to append to the end of the message
                (defaults to a newline).
            location: The location of the error (defaults to `call_location`).
        """
        comptime target_level = Level.WARNING

        comptime if not Self._is_disabled[target_level]():
            self._write_out[target_level](
                *values,
                sep=sep,
                end=end,
                location=location.or_else(call_location()),
            )

    @always_inline
    def error[
        *Ts: Writable
    ](
        self,
        *values: *Ts,
        sep: StaticString = " ",
        end: StaticString = "\n",
        location: Optional[SourceLocation] = None,
    ):
        """Logs an error message.

        Parameters:
            Ts: The types of values to log.

        Args:
            values: The values to log.
            sep: The separator to use between values (defaults to a space).
            end: The string to append to the end of the message
                (defaults to a newline).
            location: The location of the error (defaults to `call_location`).
        """
        comptime target_level = Level.ERROR

        comptime if not Self._is_disabled[target_level]():
            self._write_out[target_level](
                *values,
                sep=sep,
                end=end,
                location=location.or_else(call_location()),
            )

    @always_inline
    def critical[
        *Ts: Writable
    ](
        self,
        *values: *Ts,
        sep: StaticString = " ",
        end: StaticString = "\n",
        location: Optional[SourceLocation] = None,
    ):
        """Logs a critical message and aborts execution.

        Parameters:
            Ts: The types of values to log.

        Args:
            values: The values to log.
            sep: The separator to use between values (defaults to a space).
            end: The string to append to the end of the message
                (defaults to a newline).
            location: The location of the error (defaults to `call_location`).

        """
        comptime target_level = Level.CRITICAL

        comptime if not Self._is_disabled[target_level]():
            self._write_out[target_level](
                *values,
                sep=sep,
                end=end,
                location=location.or_else(call_location()),
            )

        abort()

    def _write_out[
        _level: Level,
        *Ts: Writable,
    ](
        self,
        *values: *Ts,
        location: SourceLocation,
        sep: StaticString = " ",
        end: StaticString = "\n",
    ):
        comptime color = _level.color()

        var file = self._fd
        var buffer = _FlushingWriteBuffer(file)

        if self._prefix:
            buffer.write(Text[color](self._prefix))
        else:
            buffer.write(Text[color](_level), "::: ")

        if self._source_location:
            buffer.write("[", location, "] ")

        comptime length = values.__len__()

        comptime for i in range(length):
            values[i].write_to(buffer)

            comptime if i < length - 1:
                buffer.write(sep)

        buffer.write(end)
        buffer.flush()
