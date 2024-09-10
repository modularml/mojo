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
"""Defines the builtin `NoneType`.

These are Mojo built-ins, so you don't need to import them.
"""


@value
@register_passable("trivial")
struct NoneType(
    CollectionElement,
    CollectionElementNew,
    Formattable,
    Representable,
    Stringable,
):
    """Represents the absence of a value."""

    alias _mlir_type = __mlir_type.`!kgen.none`
    """Raw MLIR type of the `None` value."""

    var _value: Self._mlir_type

    @always_inline
    fn __init__(inout self):
        """Construct an instance of the `None` type."""
        self._value = None

    @always_inline
    fn __init__(inout self, *, other: Self):
        """Explicit copy constructor.

        Args:
            other: Another `NoneType` instance to copy.
        """
        self._value = None

    @no_inline
    fn __str__(self) -> String:
        """Returns the string representation of `None`.

        Returns:
            `"None"`.
        """
        return "None"

    @no_inline
    fn __repr__(self) -> String:
        """Returns the string representation of `None`.

        Returns:
            `"None"`.
        """
        return "None"

    @no_inline
    fn format_to(self, inout writer: Formatter):
        """Write `None` to a formatter stream.

        Args:
            writer: The formatter to write to.
        """
        writer.write("None")
