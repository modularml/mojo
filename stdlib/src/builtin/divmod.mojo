# Currently, only integers are supported.
# We can support more types when Tuple supports non-regular types.
fn divmod(a: Int, b: Int) raises -> Tuple[Int, Int]:
    """Performs integer division and returns the quotient and the remainder.

    Currently supported only for integers. Support for more standard library types
    like Int8, Int16... is planned.

    This method calls `a.__divmod__(b)`, thus, the actual implementation of
    divmod should go in the `__divmod__` method of the struct of `a` and `b`.

    Args:
        a: The dividend.
        b: The divisor.

    Returns:
        A tuple containing the quotient and the remainder.

    Raises:
        ZeroDivisionError: If `b` is zero.
    """
    return a.__divmod__(b)
