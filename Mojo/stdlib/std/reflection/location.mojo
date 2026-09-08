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
"""Implements utilities to capture and represent source code location.

This module provides compile-time and runtime introspection of source locations:

- `SourceLocation` - A struct holding file name, line, and column information.
- `source_location()` - Returns the location where this function is called.
- `call_location()` - Returns the caller's location (for use in inlined functions).

These utilities are useful for error reporting, logging, debugging, and building
custom assertion functions that report meaningful locations to users.

Example using `source_location()` to get the current location:

```mojo
from std.reflection import source_location

def main():
    var loc = source_location()
    print(loc)  # Prints: /path/to/file.mojo:5:15
    print("Line:", loc.line(), "Column:", loc.column())
```

Example using `call_location()` for a custom assertion that reports the
caller's location. Note that `@always_inline` is required for `call_location()`
to work - the function must be inlined so the compiler can capture the caller's
location:

```mojo
from std.reflection import call_location

@always_inline  # Required for call_location() to work
def my_assert(cond: Bool, msg: String = "assertion failed") raises:
    if not cond:
        raise Error(call_location().prefix(msg))

def main() raises:
    var x = 5
    my_assert(x > 10, "x must be > 10")  # Error points to THIS line
```
"""


from std.memory import MaybeUninit
from std.utils._nicheable import UnsafeSingleNicheable
from .reflect import reflect


struct SourceLocation(TrivialRegisterPassable, UnsafeSingleNicheable, Writable):
    """Type to carry file name, line, and column information.

    This struct stores source location data and provides utilities for formatting
    location-prefixed messages, which is useful for error reporting and debugging.

    Example:

    ```mojo
    from std.reflection import source_location, SourceLocation

    def main():
        # Get current location
        var loc = source_location()
        print(loc)  # Prints: /path/to/file.mojo:6:19

        # Use prefix() for error-style messages
        print(loc.prefix("something went wrong"))
        # Prints: At /path/to/file.mojo:6:19: something went wrong

        # Access individual fields
        print("File:", loc.file_name())
        print("Line:", loc.line())
        print("Column:", loc.column())
    ```
    """

    var _line: Int
    """The line number (1-indexed)."""
    var _col: Int
    """The column number (1-indexed)."""
    var _file_name: StaticString
    """The file name."""

    @always_inline
    @doc_hidden
    def __init__(out self, line: Int, col: Int, file_name: StaticString):
        """Constructs a `SourceLocation` from line, column, and file name.

        `line` and `col` must be >= 1.

        Args:
            line: The 1-indexed line number.
            col: The 1-indexed column number.
            file_name: The file name.
        """
        # Note: Do not use `assert` or `debug_assert` here. Those internally
        # call `call_location()` which returns a `SourceLocation`, causing
        # circular elaboration that crashes the compiler.
        self._line = line
        self._col = col
        self._file_name = file_name

    @always_inline
    def line(self) -> Int:
        """Returns the 1-indexed line number.

        Returns:
            The line number.
        """
        return self._line

    @always_inline
    def column(self) -> Int:
        """Returns the 1-indexed column number.

        Returns:
            The column number.
        """
        return self._col

    @always_inline
    def file_name(self) -> StaticString:
        """Returns the file name.

        Returns:
            The file name.
        """
        return self._file_name

    @no_inline
    def prefix[T: Writable](self, msg: T) -> String:
        """Returns the given message prefixed with the source location.

        Parameters:
            T: The type of the message.

        Args:
            msg: The message to attach the prefix to.

        Returns:
            A string in the format "At file:line:col: msg".
        """
        return String(t"At {self}: {msg}")

    def write_to(self, mut writer: Some[Writer]):
        """
        Formats the source location to the provided Writer.

        Args:
            writer: The object to write to.
        """
        writer.write(self._file_name, ":", self._line, ":", self._col)

    comptime _LineNiche = -1
    comptime _LineByteOffset = reflect[Self].field_offset[name="_line"]()

    @staticmethod
    @always_inline
    @doc_hidden
    def write_niche(memory: MutPointer[MaybeUninit[Self], _]):
        memory.unsafe_bitcast[Byte]().unsafe_offset(
            Self._LineByteOffset
        ).unsafe_bitcast[Int]().write(Self._LineNiche)

    @staticmethod
    @always_inline
    @doc_hidden
    def isa_niche(memory: ImmPointer[MaybeUninit[Self], _]) -> Bool:
        return (
            memory.unsafe_bitcast[Byte]()
            .unsafe_offset(Self._LineByteOffset)
            .unsafe_bitcast[Int]()[]
            == Self._LineNiche
        )


@always_inline("nodebug")
def source_location() -> SourceLocation:
    """Returns the location for where this function is called.

    This currently doesn't work when called in a parameter expression.

    Returns:
        The location information of the `source_location()` call.

    Example:

    ```mojo
    from std.reflection import source_location

    def log_message(msg: String):
        var loc = source_location()
        print("[", loc.file_name(), ":", loc.line(), "]", msg)

    def main():
        log_message("hello")  # Prints: [ /path/to/file.mojo : 4 ] hello
    ```
    """
    var line, col, file_name = __mlir_op.`kgen.source_loc`[
        inlineCount=Int(0).__mlir_index__(),
        _type=Tuple[
            __mlir_type.index,
            __mlir_type.index,
            __mlir_type.`!kgen.string`,
        ],
    ]()

    return SourceLocation(
        Int(SIMDLength(mlir_value=line)),
        Int(SIMDLength(mlir_value=col)),
        StaticString(file_name),
    )


@always_inline("nodebug")
def call_location[*, inline_count: Int = 1]() -> SourceLocation:
    """Returns the location for where the caller of this function is called.

    An optional `inline_count` parameter can be specified to skip over that many
    levels of calling functions.

    This should only be used when enclosed in a series of `@always_inline` or
    `@always_inline("nodebug")` function calls, where the layers of calling
    functions is no fewer than `inline_count`.

    For example, when `inline_count = 1`, only the caller of this function needs
    to be `@always_inline` or `@always_inline("nodebug")`. This function will
    return the source location of the caller's invocation.

    When `inline_count = 2`, the caller of the caller of this function also
    needs to be inlined. This function will return the source location of the
    caller's caller's invocation.

    This currently doesn't work when the `inline_count`-th wrapping caller is
    called in a parameter expression.

    Parameters:
        inline_count: The number of inline call levels to skip.

    Returns:
        The location information of where the caller of this function (i.e. the
        function whose body `call_location()` is used in) is called.

    Example:

    ```mojo
    from std.reflection import call_location

    @always_inline  # Required for call_location() to work
    def assert_positive(value: Int) raises:
        # call_location() returns where assert_positive() was called,
        # not where call_location() itself is called.
        if value <= 0:
            raise Error(call_location().prefix("value must be positive"))

    def main():
        try:
            assert_positive(-1)  # Error will point to THIS line
        except e:
            print(e)
    ```
    """
    var line, col, file_name = __mlir_op.`kgen.source_loc`[
        inlineCount=inline_count.__mlir_index__(),
        _type=Tuple[
            __mlir_type.index,
            __mlir_type.index,
            __mlir_type.`!kgen.string`,
        ],
    ]()

    return SourceLocation(
        Int(SIMDLength(mlir_value=line)),
        Int(SIMDLength(mlir_value=col)),
        StaticString(file_name),
    )
