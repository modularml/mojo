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
"""Implements utilities to capture and represent source code location.
"""


@value
@register_passable("trivial")
struct _SourceLocation(Writable, Stringable):
    """Type to carry file name, line, and column information."""

    var line: Int
    var col: Int
    var file_name: StringLiteral

    fn __init__(out self, *, other: Self):
        self = other

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn prefix[T: Stringable](self, msg: T) -> String:
        """Return the given message prefixed with the pretty-printer location.

        Parameters:
            T: The type of the message.

        Args:
            msg: The message to attach the prefix to.
        """
        return "At " + str(self) + ": " + str(msg)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats the source location to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        writer.write(self.file_name, ":", self.line, ":", self.col)


@always_inline("nodebug")
fn __source_location() -> _SourceLocation:
    """Returns the location for where this function is called.

    This currently doesn't work when called in a parameter expression.

    Returns:
        The location information of the __source_location() call.
    """
    var line: __mlir_type.index
    var col: __mlir_type.index
    var file_name: __mlir_type.`!kgen.string`
    line, col, file_name = __mlir_op.`kgen.source_loc`[
        inlineCount = Int(0).value,
        _type = (
            __mlir_type.index,
            __mlir_type.index,
            __mlir_type.`!kgen.string`,
        ),
    ]()

    return _SourceLocation(line, col, file_name)


@always_inline("nodebug")
fn __call_location[inline_count: Int = 1]() -> _SourceLocation:
    """Returns the location for where the caller of this function is called. An
    optional `inline_count` parameter can be specified to skip over that many
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

    Returns:
        The location information of where the caller of this function (i.e. the
          function whose body __call_location() is used in) is called.
    """
    var line: __mlir_type.index
    var col: __mlir_type.index
    var file_name: __mlir_type.`!kgen.string`
    line, col, file_name = __mlir_op.`kgen.source_loc`[
        inlineCount = inline_count.value,
        _type = (
            __mlir_type.index,
            __mlir_type.index,
            __mlir_type.`!kgen.string`,
        ),
    ]()

    return _SourceLocation(line, col, file_name)
