# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Defines some type aliases.

These are Mojo built-ins, so you don't need to import them.
"""

alias AnyRegType = __mlir_type.`!kgen.type`
"""Represents any register passable Mojo data type."""

alias NoneType = __mlir_type.`!kgen.none`
"""Represents the absence of a value."""

alias ImmLifetime = __mlir_type.`!lit.lifetime<0>`
"""Immutable lifetime reference."""

alias MutLifetime = __mlir_type.`!lit.lifetime<1>`
"""Mutable lifetime reference."""


# Helper to build !lit.lifetime type.
# TODO: Should be a parametric alias.
# TODO: Should take a Bool, not an i1.
struct AnyLifetime[is_mutable: __mlir_type.i1]:
    """This represents a lifetime reference of potentially parametric type.
    TODO: This should be replaced with a parametric type alias.

    Parameters:
        is_mutable: Whether the lifetime reference is mutable.
    """

    alias type = __mlir_type[
        `!lit.lifetime<`,
        is_mutable,
        `>`,
    ]
