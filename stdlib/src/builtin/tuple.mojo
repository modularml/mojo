# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the Tuple type.

These are Mojo built-ins, so you don't need to import them.
"""

from debug.lldb import lldb_formatter_wrapping_type

# ===----------------------------------------------------------------------===#
# Tuple
# ===----------------------------------------------------------------------===#


@lldb_formatter_wrapping_type
@register_passable
struct Tuple[*Ts: AnyRegType](Sized, CollectionElement):
    """The type of a literal tuple expression.

    A tuple consists of zero or more values, separated by commas.

    Parameters:
        Ts: The elements type.
    """

    var storage: __mlir_type[`!kgen.pack<`, Ts, `>`]
    """The underlying storage for the tuple."""

    @always_inline("nodebug")
    fn __init__(*args: *Ts) -> Self:
        """Construct the tuple.

        Args:
            args: Initial values.

        Returns:
            Constructed tuple.
        """
        return Self {storage: args}

    @always_inline("nodebug")
    fn __copyinit__(existing: Self) -> Self:
        """Copy construct the tuple.

        Returns:
            Constructed tuple.
        """
        return Self {storage: existing.storage}

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the number of elements in the tuple.

        Returns:
            The tuple length.
        """
        return __mlir_op.`pop.variadic.size`(Ts)

    @always_inline("nodebug")
    fn get[i: Int, T: AnyRegType](self) -> T:
        """Get a tuple element.

        Parameters:
            i: The element index.
            T: The element type.

        Returns:
            The tuple element at the requested index.
        """
        return __mlir_op.`kgen.pack.get`[_type=T, index = i.value](self.storage)
