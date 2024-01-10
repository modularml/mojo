# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Defines some type aliases.

These are Mojo built-ins, so you don't need to import them.
"""

alias AnyRegType = __mlir_type.`!kgen.anyregtype`
"""Represents any register passable Mojo data type."""

alias NoneType = __mlir_type.`!kgen.none`
"""Represents the absence of a value."""

alias Lifetime = __mlir_type.`!lit.lifetime`
"""Value lifetime specifier."""
