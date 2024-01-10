# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements type rebind.

These are Mojo built-ins, so you don't need to import them.
"""


@always_inline("nodebug")
fn rebind[
    dest_type: AnyRegType,
    src_type: AnyRegType,
](val: src_type) -> dest_type:
    """Statically assert that a parameter input type `src_type` resolves to the
    same type as a parameter result type `dest_type` after function
    instantiation and "rebind" the input to the result type.

    This function is meant to be used in uncommon cases where a parametric type
    depends on the value of a constrained parameter in order to manually refine
    the type with the constrained parameter value.

    Parameters:
        dest_type: The type to rebind to.
        src_type: The original type.

    Args:
        val: The value to rebind.

    Returns:
        The rebound value of `dest_type`.
    """
    return __mlir_op.`kgen.rebind`[_type=dest_type](val)
