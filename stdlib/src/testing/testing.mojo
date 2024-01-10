# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements various testing utils.

You can import these APIs from the `testing` package. For example:

```mojo
from testing import assert_true
```
"""

from math import abs, isclose


fn assert_true(val: object, msg: String = "") raises:
    """Asserts that the input value is True. If it is not then an
    Error is raised.

    Args:
        val: The value to assert to be True.
        msg: The message to be printed if the assertion fails.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if not val:
        raise Error("AssertionError: " + msg)


fn assert_true(val: Bool, msg: String = "") raises:
    """Asserts that the input value is True. If it is not then an Error
    is raised.

    Args:
        val: The value to assert to be True.
        msg: The message to be printed if the assertion fails.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if not val:
        raise Error("AssertionError: " + msg)


fn assert_false(val: object, msg: String = "") raises:
    """Asserts that the input value is False. If it is not then an Error is
    raised.

    Args:
        val: The value to assert to be False.
        msg: The message to be printed if the assertion fails.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if val:
        raise Error("AssertionError: " + msg)


fn assert_false(val: Bool, msg: String = "") raises:
    """Asserts that the input value is False. If it is not then an Error is
    raised.

    Args:
        val: The value to assert to be False.
        msg: The message to be printed if the assertion fails.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if val:
        raise Error("AssertionError: " + msg)


# TODO: Collapse these two overloads for generic T that has the
# Equality Comparable trait.
fn assert_equal(lhs: Int, rhs: Int) raises:
    """Asserts that the input values are equal. If it is not then an Error
    is raised.

    Args:
        lhs: The lhs of the equality.
        rhs: The rhs of the equality.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs != rhs:
        raise Error(
            "AssertionError: " + String(lhs) + " is not equal to " + String(rhs)
        )


fn assert_equal(lhs: String, rhs: String) raises:
    """Asserts that the input values are equal. If it is not then an Error
    is raised.

    Args:
        lhs: The lhs of the equality.
        rhs: The rhs of the equality.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs != rhs:
        raise Error("AssertionError: " + lhs + " is not equal to " + rhs)


fn assert_equal[
    type: DType, size: Int
](lhs: SIMD[type, size], rhs: SIMD[type, size]) raises:
    """Asserts that the input values are equal. If it is not then an
    Error is raised.

    Parameters:
        type: The dtype of the left- and right-hand-side SIMD vectors.
        size: The width of the left- and right-hand-side SIMD vectors.

    Args:
        lhs: The lhs of the equality.
        rhs: The rhs of the equality.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs != rhs:
        raise Error(
            "AssertionError: " + String(lhs) + " is not equal to " + String(rhs)
        )


fn assert_not_equal(lhs: Int, rhs: Int) raises:
    """Asserts that the input values are not equal. If it is not then an
    Error is raised.

    Args:
        lhs: The lhs of the inequality.
        rhs: The rhs of the inequality.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs == rhs:
        raise Error(
            "AssertionError: " + String(lhs) + " is not equal to " + String(rhs)
        )


fn assert_not_equal(lhs: String, rhs: String) raises:
    """Asserts that the input values are not equal. If it is not then an
    an Error is raised.

    Args:
        lhs: The lhs of the inequality.
        rhs: The rhs of the inequality.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs == rhs:
        raise Error("AssertionError: " + lhs + " is not equal to " + rhs)


fn assert_not_equal[
    type: DType, size: Int
](lhs: SIMD[type, size], rhs: SIMD[type, size]) raises:
    """Asserts that the input values are not equal. If it is not then an
    Error is raised.

    Parameters:
        type: The dtype of the left- and right-hand-side SIMD vectors.
        size: The width of the left- and right-hand-side SIMD vectors.

    Args:
        lhs: The lhs of the inequality.
        rhs: The rhs of the inequality.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs == rhs:
        raise Error(
            "AssertionError: " + String(lhs) + " is not equal to " + String(rhs)
        )


fn assert_almost_equal[
    type: DType, size: Int
](
    lhs: SIMD[type, size],
    rhs: SIMD[type, size],
    absolute_tolerance: SIMD[type, 1] = 1e-08,
    relative_tolerance: SIMD[type, 1] = 1e-05,
) raises:
    """Asserts that the input values are equal up to a tolerance. If it is
    not then an Error is raised.

    Parameters:
        type: The dtype of the left- and right-hand-side SIMD vectors.
        size: The width of the left- and right-hand-side SIMD vectors.

    Args:
        lhs: The lhs of the equality.
        rhs: The rhs of the equality.
        absolute_tolerance: The absolute tolerance.
        relative_tolerance: The relative tolerance.

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    let almost_equal = isclose(
        lhs, rhs, absolute_tolerance, relative_tolerance
    ).reduce_and()
    if not almost_equal:
        raise Error(
            "AssertionError: "
            + String(lhs)
            + " is not close to "
            + String(rhs)
            + " with a diff of "
            + abs(lhs - rhs)
        )
