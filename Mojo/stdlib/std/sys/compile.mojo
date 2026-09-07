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
"""Implements functions that return compile-time information.
"""
from .defines import get_defined_int, get_defined_string, is_defined
from std.collections.string.string_span import _get_kgen_string

# ===----------------------------------------------------------------------=== #
# codegen_reachable
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
def codegen_unreachable[cond: Bool, msg: StaticString, *extra: StaticString]():
    """Compilation fails if cond is True and the caller of the function
    is being generated as runtime code.

    Parameters:
        cond: The bool value for reachability.
        msg: The message to display on failure.
        extra: Additional messages to concatenate to msg.

    """
    __mlir_op.`kgen.codegen.reachable`[
        cond=(not cond).__mlir_i1__(),
        message=_get_kgen_string[msg, *extra](),
        _type=None,
    ]()


# ===----------------------------------------------------------------------=== #
# OptimizationLevel
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _OptimizationLevel(ImplicitlyCopyable, Intable, Writable):
    """Represents the optimization level used during compilation.

    The optimization level is determined by the __OPTIMIZATION_LEVEL environment
    variable, with a default value of 4 if not specified.

    Attributes:
        level: The integer value of the optimization level.
    """

    comptime level = get_defined_int["__OPTIMIZATION_LEVEL", 4]()

    def __int__(self) -> Int:
        """Returns the integer value of the optimization level.

        Returns:
            The optimization level as an integer.
        """
        return Self.level

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes the optimization level to a writer."""
        writer.write(Self.level)


comptime OptimizationLevel = _OptimizationLevel()
"""Represents the optimization level used during compilation."""

# ===----------------------------------------------------------------------=== #
# DebugLevel
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _DebugLevel(ImplicitlyCopyable, Writable):
    """Represents the debug level used during compilation.

    The debug level is determined by the __DEBUG_LEVEL environment variable,
    with a default value of "none" if not specified.

    Attributes:
        level: The string value of the debug level.
    """

    comptime level = get_defined_string["__DEBUG_LEVEL", "none"]()

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes the optimization level to a writer."""
        writer.write(Self.level)


comptime DebugLevel = _DebugLevel()
"""Represents the debug level used during compilation."""

# ===----------------------------------------------------------------------=== #
# SanitizeAddress
# ===----------------------------------------------------------------------=== #

comptime SanitizeAddress = is_defined[
    "__SANITIZE_ADDRESS"
]() and get_defined_int["__SANITIZE_ADDRESS"]() == 1
"""True if address sanitizer is enabled at compile-time."""
