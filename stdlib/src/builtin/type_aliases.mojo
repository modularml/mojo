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

from builtin.builtin_list import _lit_mut_cast

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


struct _origin_inner[is_mutable: Bool, //]:
    alias type = __mlir_type[
        `!lit.origin<`,
        is_mutable.value,
        `>`,
    ]


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
    """The underlying MLIR type."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var value: Self.type
    """The underlying MLIR value."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_private
    @implicit
    @always_inline("nodebug")
    fn __init__(out self, mlir_origin: Self.type):
        """Initialize an Origin from an MLIR origin value.

        Args:
            mlir_origin: The MLIR origin value.
        """
        self.value = mlir_origin

    # FIXME: This will be useful to do `Origin[False].coerce[mlir_origin]()`
    # and to replace _lit_mut_cast_origin. But:
    # error: invalid call to 'coerce': callee parameter #1 has 'Bool'
    # type, but value has type 'Origin[is_mutable.value]'
    # @always_inline("nodebug")
    # @staticmethod
    # fn coerce[mutable: Bool, //, origin: Origin[mutable].type]() -> Self.type:
    #     """Return a coerced version of the Origin.

    #     Returns:
    #         A mutable version of the Origin.
    #     """
    #     alias result = __mlir_attr[
    #     `#lit.origin.mutcast<`,
    #     origin,
    #     `> : !lit.origin<`,
    #     +Self.is_mutable.value,
    #     `>`,
    #     ]
    #     return result
