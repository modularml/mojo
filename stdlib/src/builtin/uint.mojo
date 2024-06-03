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
"""Implements the UInt class.

These are Mojo built-ins, so you don't need to import them.
"""


@lldb_formatter_wrapping_type
@value
@register_passable("trivial")
struct UInt:
    """This type represents an unsigned integer.

    The size of this unsigned integer is platform-dependent.

    If you wish to use a fixed size unsigned integer, consider using
    `UInt8`, `UInt16`, `UInt32`, or `UInt64`.
    """

    var value: __mlir_type.index
    """The underlying storage for the integer value.

    Note that it is the same type as the `Int.value` field.
    MLIR doesn't differentiate between signed and unsigned integers
    when it comes to storing them with the index dialect. 
    The difference is in the operations that are performed on them,
    which have signed and unsigned variants.
    """

    @always_inline("nodebug")
    fn __init__(inout self):
        """Default constructor that produces zero."""
        self.value = __mlir_op.`index.constant`[value = __mlir_attr.`0:index`]()

    @always_inline("nodebug")
    fn __init__(inout self, value: __mlir_type.index):
        """Construct Int from the given index value.

        Args:
            value: The init value.
        """
        self.value = value

    @always_inline("nodebug")
    fn __init__(inout self, value: IntLiteral):
        """Construct Int from the given IntLiteral value.

        Args:
            value: The init value.
        """
        # TODO: Find a way to convert directly without using UInt64.
        # This is because the existing
        # __mlir_op.`kgen.int_literal.convert`
        # in IntLiteral.__as_mlir_index()
        # assumes that the index represents an unsigned integer.
        # We need a variant for signed integers.
        self.value = int(UInt64(value)).value

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Convert this Int to a string.

        Returns:
            The string representation of this Int.
        """
        return str(UInt64(self))
