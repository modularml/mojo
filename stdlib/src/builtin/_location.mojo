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
struct _SourceLocation(Stringable):
    """Type to carry file name, line, and column information."""

    var line: Int
    var col: Int
    var file_name: StringLiteral

    fn __str__(self) -> String:
        return str(self.file_name) + ":" + str(self.line) + ":" + str(self.col)

    fn prefix[T: Stringable](self, msg: T) -> String:
        """Return the given message prefixed with the pretty-printer location.

        Parameters:
            T: The type of the message.

        Args:
            msg: The message to attach the prefix to.
        """
        return "At " + str(self) + ": " + str(msg)


@always_inline("nodebug")
fn __source_location() -> _SourceLocation:
    """Returns the location where it's called.

    This currently doesn't work when called in a parameter expression.

    Returns:
        The location information of the __source_location() call.
    """
    var line: __mlir_type.index
    var col: __mlir_type.index
    var file_name: __mlir_type.`!kgen.string`
    line, col, file_name = __mlir_op.`kgen.source_loc`[
        _properties = __mlir_attr.`{inlineCount = 0 : i64}`,
        _type = (
            __mlir_type.index,
            __mlir_type.index,
            __mlir_type.`!kgen.string`,
        ),
    ]()

    return _SourceLocation(line, col, file_name)


@always_inline("nodebug")
fn __call_location() -> _SourceLocation:
    """Returns the location where the enclosing function is called.

    This should only be used in `@always_inline` or `@always_inline("nodebug")`
    functions so that it returns the source location of where the enclosing
    function is called at (even if inside another `@always_inline("nodebug")`
    function).

    This currently doesn't work when this or the enclosing function is called in
    a parameter expression.

    Returns:
        The location information of where the enclosing function (i.e. the
          function whose body __call_location() is used in) is called.
    """
    var line: __mlir_type.index
    var col: __mlir_type.index
    var file_name: __mlir_type.`!kgen.string`
    line, col, file_name = __mlir_op.`kgen.source_loc`[
        _properties = __mlir_attr.`{inlineCount = 1 : i64}`,
        _type = (
            __mlir_type.index,
            __mlir_type.index,
            __mlir_type.`!kgen.string`,
        ),
    ]()

    return _SourceLocation(line, col, file_name)
