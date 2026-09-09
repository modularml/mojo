# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Implements various testing utils.

You can import these APIs from the `testing` package. For example:

```mojo
from std.testing import assert_true

def main() raises:
    x = 1
    y = 2
    try:
        assert_true(x==1)
        assert_true(y==2)
        assert_true((x+y)==3)
        print("All assertions succeeded")
    except e:
        print("At least one assertion failed:")
        print(e)
```
"""

from std.math import isclose

from std.reflection import call_location, SourceLocation
from std.memory import unsafe_memcmp
from std.python import PythonObject
from std.utils._ansi import Color, Text

# ===----------------------------------------------------------------------=== #
# Assertions
# ===----------------------------------------------------------------------=== #


@always_inline
def _assert_error[T: Writable](msg: T, loc: SourceLocation) -> Error:
    return Error(loc.prefix(t"AssertionError: {msg}"))


@always_inline
def assert_true[
    T: Boolable, //
](
    val: T,
    msg: String = "condition was unexpectedly False",
    *,
    location: Optional[SourceLocation] = None,
) raises:
    """Asserts that the input value is True and raises an Error if it's not.

    Parameters:
        T: The type of the value argument.

    Args:
        val: The value to assert to be True.
        msg: The message to be printed if the assertion fails.
        location: The location of the error (defaults to `call_location`).

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if not val:
        raise _assert_error(msg, location.or_else(call_location()))


@always_inline
def assert_false[
    T: Boolable, //
](
    val: T,
    msg: String = "condition was unexpectedly True",
    *,
    location: Optional[SourceLocation] = None,
) raises:
    """Asserts that the input value is False and raises an Error if it's not.

    Parameters:
        T: The type of the value argument.

    Args:
        val: The value to assert to be False.
        msg: The message to be printed if the assertion fails.
        location: The location of the error (defaults to `call_location`).

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if val:
        raise _assert_error(msg, location.or_else(call_location()))


@always_inline
def assert_equal[
    T: Equatable & Writable,
    //,
](
    lhs: T,
    rhs: T,
    msg: String = "",
    *,
    location: Optional[SourceLocation] = None,
) raises:
    """Asserts that the input values are equal. If it is not then an Error
    is raised.

    Parameters:
        T: The type of the input values.

    Args:
        lhs: The lhs of the equality.
        rhs: The rhs of the equality.
        msg: The message to be printed if the assertion fails.
        location: The location of the error (defaults to `call_location`).

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs != rhs:
        raise _assert_cmp_error["`left == right` comparison"](
            String(lhs),
            String(rhs),
            msg=msg,
            loc=location.or_else(call_location()),
        )


# TODO: Remove the PythonObject, String and List overloads once we have
# more powerful traits.


# TODO(MSTDL-1071):
#   Once Mojo supports parametric traits, implement Equatable for
#   StringSlice such that string slices with different origin types can be
#   compared, then drop this overload.
@always_inline
def assert_equal[
    O1: ImmOrigin, O2: ImmOrigin
](
    lhs: List[StringSlice[O1]],
    rhs: List[StringSlice[O2]],
    msg: String = "",
    *,
    location: Optional[SourceLocation] = None,
) raises:
    """Asserts that two lists are equal.

    Parameters:
        O1: The origin of lhs.
        O2: The origin of rhs.

    Args:
        lhs: The left-hand side list.
        rhs: The right-hand side list.
        msg: The message to be printed if the assertion fails.
        location: The location of the error (defaults to `call_location`).

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """

    # Cast `rhs` to have the same origin as `lhs`, so that we can delegate to
    # `List.__ne__`.
    var rhs_origin_casted = rebind[List[StringSlice[O1]]](rhs).copy()

    if lhs != rhs_origin_casted:
        raise _assert_cmp_error["`left == right` comparison"](
            String(lhs),
            String(rhs),
            msg=msg,
            loc=location.or_else(call_location()),
        )


@always_inline
def assert_equal(
    lhs: StringSlice[mut=False, _],
    rhs: StringSlice[mut=False, _],
    msg: String = "",
    *,
    location: Optional[SourceLocation] = None,
) raises:
    """Asserts that a `StringSlice` is equal to a `String`.

    Args:
        lhs: The left-hand side value.
        rhs: The right-hand side value.
        msg: An optional custom error message.
        location: The source location of the assertion (defaults to caller location).

    Raises:
        If the values are not equal.
    """
    if lhs != rhs:
        raise _assert_cmp_error["`left == right` comparison"](
            String(lhs),
            String(rhs),
            msg=msg,
            loc=location.or_else(call_location()),
        )


@always_inline
def assert_equal_pyobj(
    lhs: PythonObject,
    rhs: PythonObject,
    msg: String = "",
    *,
    location: Optional[SourceLocation] = None,
) raises:
    """Asserts that the `PythonObject`s are equal. If it is not then an Error
    is raised.

    Args:
        lhs: The lhs of the equality.
        rhs: The rhs of the equality.
        msg: The message to be printed if the assertion fails.
        location: The location of the error (default to the `call_location`).

    Raises:
        An Error with the provided message if assert fails.
    """
    if lhs != rhs:
        raise _assert_cmp_error["`left == right` comparison"](
            String(lhs),
            String(rhs),
            msg=msg,
            loc=location.or_else(call_location()),
        )


@always_inline
def assert_not_equal[
    T: Equatable & Writable,
    //,
](
    lhs: T,
    rhs: T,
    msg: String = "",
    *,
    location: Optional[SourceLocation] = None,
) raises:
    """Asserts that the input values are not equal. If it is not then an
    Error is raised.

    Parameters:
        T: The type of the input values.

    Args:
        lhs: The lhs of the inequality.
        rhs: The rhs of the inequality.
        msg: The message to be printed if the assertion fails.
        location: The location of the error (defaults to `call_location`).

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs == rhs:
        raise _assert_cmp_error["`left != right` comparison"](
            String(lhs),
            String(rhs),
            msg=msg,
            loc=location.or_else(call_location()),
        )


@always_inline
def assert_almost_equal[
    dtype: DType, size: SIMDLength
](
    lhs: SIMD[dtype, size],
    rhs: SIMD[dtype, size],
    msg: String = "",
    *,
    atol: Float64 = 1e-08,
    rtol: Float64 = 1e-05,
    equal_nan: Bool = False,
    location: Optional[SourceLocation] = None,
) raises:
    """Asserts that the input values are equal up to a tolerance. If it is
    not then an Error is raised.

    When the type is boolean or integral, then equality is checked. When the
    type is floating-point, then this checks if the two input values are
    numerically the close using the $abs(lhs - rhs) <= max(rtol * max(abs(lhs),
    abs(rhs)), atol)$ formula.

    Constraints:
        The type must be boolean, integral, or floating-point.

    Parameters:
        dtype: The dtype of the left- and right-hand-side SIMD vectors.
        size: The width of the left- and right-hand-side SIMD vectors.

    Args:
        lhs: The lhs of the equality.
        rhs: The rhs of the equality.
        msg: The message to print.
        atol: The absolute tolerance.
        rtol: The relative tolerance.
        equal_nan: Whether to treat nans as equal.
        location: The location of the error (defaults to `call_location`).

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    comptime assert (
        dtype == .bool or dtype.is_integral() or dtype.is_floating_point()
    ), "type must be boolean, integral, or floating-point"

    var almost_equal = isclose(
        lhs, rhs, atol=atol, rtol=rtol, equal_nan=equal_nan
    )

    if not all(almost_equal):
        var err = String(lhs, " is not close to ", rhs)

        comptime if dtype.is_integral() or dtype.is_floating_point():
            err += String(" with a diff of ", abs(lhs - rhs))

        if msg:
            err += String(" (", msg, ")")

        raise _assert_error(err, location.or_else(call_location()))


@always_inline
def assert_is[
    T: Identifiable & Writable, //
](
    lhs: T,
    rhs: T,
    msg: String = "",
    *,
    location: Optional[SourceLocation] = None,
) raises:
    """Asserts that the input values have the same identity. If they do not
    then an Error is raised.

    Parameters:
        T: A Writable and Identifiable type.

    Args:
        lhs: The lhs of the `is` statement.
        rhs: The rhs of the `is` statement.
        msg: The message to be printed if the assertion fails.
        location: The location of the error (defaults to `call_location`).

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs is not rhs:
        raise _assert_cmp_error["`left is right` identification"](
            String(lhs),
            String(rhs),
            msg=msg,
            loc=location.or_else(call_location()),
        )


@always_inline
def assert_is_not[
    T: Identifiable & Writable, //
](
    lhs: T,
    rhs: T,
    msg: String = "",
    *,
    location: Optional[SourceLocation] = None,
) raises:
    """Asserts that the input values have different identities. If they do not
    then an Error is raised.

    Parameters:
        T: A Writable and Identifiable type.

    Args:
        lhs: The lhs of the `is not` statement.
        rhs: The rhs of the `is not` statement.
        msg: The message to be printed if the assertion fails.
        location: The location of the error (defaults to `call_location`).

    Raises:
        An Error with the provided message if assert fails and `None` otherwise.
    """
    if lhs is rhs:
        raise _assert_cmp_error["`left is not right` identification"](
            String(lhs),
            String(rhs),
            msg=msg,
            loc=location.or_else(call_location()),
        )


def _colorize_diff_string[color: Color](s: String, other: String) -> String:
    """Colorizes a string by highlighting codepoints that differ from another string.

    Parameters:
        color: The color to use for highlighting differences.

    Args:
        s: The string to colorize.
        other: The string to compare against.

    Returns:
        A string with differences highlighted in the specified color.
    """
    var result = String()
    var other_codepoints = other.codepoints()
    for s_codepoint in s.codepoints():
        var other_codepoint = other_codepoints.next()
        if other_codepoint and s_codepoint == other_codepoint.value():
            # Codepoints match - no color
            result.append(s_codepoint)
        else:
            # Codepoint differs or other string is shorter - apply color
            result += String(Text[color](s_codepoint))
    return result


def _create_colored_diff(lhs: String, rhs: String) -> String:
    """Creates a colored character-by-character diff of two strings.

    Highlights differences in red for the left string and green for the right string.

    Args:
        lhs: The left-hand side string.
        rhs: The right-hand side string.

    Returns:
        A string containing the colored diff output.
    """
    return String(
        "\n   left: ",
        _colorize_diff_string[Color.RED](lhs, rhs),
        "\n  right: ",
        _colorize_diff_string[Color.GREEN](rhs, lhs),
    )


def _assert_cmp_error[
    cmp: String
](lhs: String, rhs: String, *, msg: String, loc: SourceLocation) -> Error:
    var err = cmp + " failed:"

    # For string comparisons, show colored diff
    err += _create_colored_diff(lhs, rhs)

    if msg:
        err += "\n  reason: " + msg
    return _assert_error(err, loc)


struct assert_raises:
    """Context manager that asserts that the block raises an exception.

    You can use this to test expected error cases, and to test that the correct
    errors are raised. Works with `Error` and any custom `Writable` error type.
    For instance:

    ```mojo
    from std.testing import assert_raises

    # Good! Caught the raised error, test passes
    with assert_raises():
        raise Error("SomeError")

    # Also good!
    with assert_raises(contains="Some"):
        raise Error("SomeError")

    # This will assert, we didn't raise
    with assert_raises():
        pass

    # This will let the underlying error propagate, failing the test
    with assert_raises(contains="Some"):
        raise Error("OtherError")
    ```
    """

    var message_contains: Optional[String]
    """If present, check that the error message contains this literal string."""

    var call_location: SourceLocation
    """Assigned the value returned by call_locations() at Self.__init__."""

    @always_inline
    def __init__(out self, *, location: Optional[SourceLocation] = None):
        """Construct a context manager with no message pattern.

        Args:
            location: The location of the error (defaults to `call_location`).
        """
        self.message_contains = None
        self.call_location = location.or_else(call_location())

    @always_inline
    def __init__(
        out self,
        *,
        contains: String,
        location: Optional[SourceLocation] = None,
    ):
        """Construct a context manager matching specific errors.

        Args:
            contains: The test will only pass if the error message
                includes the literal text passed.
            location: The location of the error (defaults to `call_location`).
        """
        self.message_contains = contains
        self.call_location = location.or_else(call_location())

    def __enter__(self):
        """Enter the context manager."""
        pass

    def __exit__(self) raises:
        """Exit the context manager with no error.

        Raises:
            AssertionError: Always. The block must raise to pass the test.
        """
        raise Error("AssertionError: Didn't raise at ", self.call_location)

    def __exit__[E: AnyType](self, error: E) raises -> Bool:
        """Exit the context manager with an error.

        Works with `Error` and any custom `Writable` error type. When
        `contains` is specified, the error type must be `Writable` so
        it can be converted to a string for matching.

        Parameters:
            E: The error type.

        Args:
            error: The error raised.

        Raises:
            Error: If `contains` is set and the error message doesn't
                include the expected string, or if `contains` is set
                but the error type is not `Writable`.

        Returns:
            True if the error was successfully caught and matched.
        """
        if self.message_contains:
            comptime assert conforms_to(
                E, Writable
            ), "assert_raises(contains=...) requires a Writable error type"
            return self.message_contains.value() in String(error)
        return True
