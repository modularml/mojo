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
"""Defines some type aliases.

These are Mojo built-ins, so you don't need to import them.
"""

alias AnyTrivialRegType = __mlir_type.`!kgen.type`
"""Represents any register passable Mojo data type."""

alias ImmutableOrigin = __mlir_type.`!lit.origin<0>`
"""Immutable origin reference type."""

alias MutableOrigin = __mlir_type.`!lit.origin<1>`
"""Mutable origin reference type."""

alias ImmutableAnyOrigin = __mlir_attr.`#lit.any.origin : !lit.origin<0>`
"""The immutable origin that might access any memory value."""

alias MutableAnyOrigin = __mlir_attr.`#lit.any.origin : !lit.origin<1>`
"""The mutable origin that might access any memory value."""

# Static constants are a named subset of the global origin.
alias StaticConstantOrigin = __mlir_attr[
    `#lit.origin.field<`,
    `#lit.static.origin : !lit.origin<0>`,
    `, "__constants__"> : !lit.origin<0>`,
]
"""An origin for strings and other always-immutable static constants."""

alias OriginSet = __mlir_type.`!lit.origin.set`
"""A set of origin parameters."""


@value
@register_passable("trivial")
struct Origin[is_mutable: Bool]:
    """This represents a origin reference for a memory value.

    Parameters:
        is_mutable: Whether the origin is mutable.
    """

    alias type = __mlir_type[
        `!lit.origin<`,
        is_mutable.value,
        `>`,
    ]

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _mlir_origin: Self.type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    # NOTE:
    #   Needs to be @implicit convertible for the time being so that
    #   `__origin_of(..)` can implicilty convert to `Origin` in use cases like:
    #       Span[Byte, __origin_of(self)]
    @implicit
    @always_inline("nodebug")
    fn __init__(out self, mlir_origin: Self.type):
        """Initialize an Origin from a raw MLIR `!lit.origin` value.

        Args:
            mlir_origin: The raw MLIR origin value."""
        self._mlir_origin = mlir_origin
