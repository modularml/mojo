# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the source range struct.
"""


@register_passable("trivial")
struct _SourceRange(Stringable):
    var file_name: StringLiteral
    var line: Int
    var col_start: Int
    var col_end: Int

    fn __init__(
        file_name: StringLiteral,
        line: Int,
        col_start: Int = -1,
        col_end: Int = -1,
    ) -> Self:
        return Self {
            file_name: file_name,
            line: line,
            col_start: col_start,
            col_end: col_end,
        }

    fn __str__(self) -> String:
        var res = String(self.file_name) + ":" + str(self.line)
        if self.col_start == -1:
            return res
        res += ":" + str(self.col_start)
        if self.col_end == -1:
            return res
        return res + ":" + str(self.col_end)
