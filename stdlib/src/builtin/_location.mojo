# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the source range struct.
"""


@value
@register_passable("trivial")
struct _SourceLocation(Stringable):
    var file_name: StringLiteral
    var function_name: StringLiteral
    var line: Int

    fn __str__(self) -> String:
        return (
            str(self.file_name)
            + ":"
            + str(self.function_name)
            + ":"
            + str(self.line)
        )
