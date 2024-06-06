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
"""Implements slice.

These are Mojo built-ins, so you don't need to import them.
"""

from sys.intrinsics import _mlirtype_is_eq
from collections import OptionalReg


@always_inline("nodebug")
fn _int_max_value() -> Int:
    # FIXME: The `slice` type should have the concept of `None` indices, but the
    # effect of a `None` end index is the same as a very large end index.
    return int(Int32.MAX)


@always_inline("nodebug")
fn _default_or[T: AnyTrivialRegType](value: T, default: Int) -> Int:
    # TODO: Handle `__index__` for other types when we have traits!
    @parameter
    if _mlirtype_is_eq[T, Int]():
        return __mlir_op.`kgen.rebind`[_type=Int](value)
    else:
        __mlir_op.`kgen.param.assert`[
            cond = (_mlirtype_is_eq[T, NoneType]()).__mlir_i1__(),
            message = "expected Int or NoneType".value,
        ]()
        return default


@register_passable("trivial")
struct Slice(Stringable, EqualityComparable):
    """Represents a slice expression.

    Objects of this type are generated when slice syntax is used within square
    brackets, e.g.:

    ```mojo
    var msg: String = "Hello Mojo"

    # Both are equivalent and print "Mojo".
    print(msg[6:])
    print(msg.__getitem__(Slice(6, len(msg))))
    ```
    """

    var start: Int
    """The starting index of the slice."""
    var end: Int
    """The end index of the slice."""
    var step: Int
    """The step increment value of the slice."""

    @always_inline("nodebug")
    fn __init__(inout self, start: Int, end: Int):
        """Construct slice given the start and end values.

        Args:
            start: The start value.
            end: The end value.
        """
        self.start = start
        self.end = end
        self.step = 1

    @always_inline("nodebug")
    fn __init__[
        T0: AnyTrivialRegType, T1: AnyTrivialRegType, T2: AnyTrivialRegType
    ](inout self, start: T0, end: T1, step: T2):
        """Construct slice given the start, end and step values.

        Parameters:
            T0: Type of the start value.
            T1: Type of the end value.
            T2: Type of the step value.

        Args:
            start: The start value.
            end: The end value.
            step: The step value.
        """
        self.start = _default_or(start, 0)
        self.end = _default_or(end, _int_max_value())
        self.step = _default_or(step, 1)

    fn __str__(self) -> String:
        """Gets the string representation of the span.

        Returns:
            The string representation of the span.
        """
        var res = str(self.start)
        res += ":"
        if self._has_end():
            res += str(self.end)
        res += ":"
        res += str(self.step)
        return res

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Compare this slice to the other.

        Args:
            other: The slice to compare to.

        Returns:
            True if start, end, and step values of this slice match the
            corresponding values of the other slice and False otherwise.
        """
        return (
            self.start == other.start
            and self.end == other.end
            and self.step == other.step
        )

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Compare this slice to the other.

        Args:
            other: The slice to compare to.

        Returns:
            False if start, end, and step values of this slice match the
            corresponding values of the other slice and True otherwise.
        """
        return not (self == other)

    @always_inline
    fn unsafe_indices(self) -> Int:
        """Return the length of the slice.

        Only use this function if start/end is guaranteed to be not None.

        Returns:
            The length of the slice.
        """

        return len(range(self.start, self.end, self.step))

    @always_inline
    fn __getitem__(self, idx: Int) -> Int:
        """Get the slice index.

        Args:
            idx: The index.

        Returns:
            The slice index.
        """
        return self.start + index(idx) * self.step

    @always_inline("nodebug")
    fn _has_end(self) -> Bool:
        return self.end != _int_max_value()


@always_inline("nodebug")
fn slice(end: Int) -> Slice:
    """Construct slice given the end value.

    Args:
        end: The end value.

    Returns:
        The constructed slice.
    """
    return Slice(0, end)


@always_inline("nodebug")
fn slice(start: Int, end: Int) -> Slice:
    """Construct slice given the start and end values.

    Args:
        start: The start value.
        end: The end value.

    Returns:
        The constructed slice.
    """
    return Slice(start, end)


# TODO(30496): Modernize the slice type
@always_inline("nodebug")
fn slice[
    T0: AnyTrivialRegType, T1: AnyTrivialRegType, T2: AnyTrivialRegType
](start: T0, end: T1, step: T2) -> Slice:
    """Construct a Slice given the start, end and step values.

    Parameters:
        T0: Type of the start value.
        T1: Type of the end value.
        T2: Type of the step value.

    Args:
        start: The start value.
        end: The end value.
        step: The step value.

    Returns:
        The constructed slice.
    """
    return Slice(start, end, step)


# ===-------------------------------------------------------------------===#
# SliceNew
# ===-------------------------------------------------------------------===#


@always_inline("nodebug")
fn _compare_optional(x: OptionalReg[Int], y: OptionalReg[Int]) -> Bool:
    if x and y:
        return x.value() == y.value()
    return not x and not y


@register_passable("trivial")
struct SliceNew(Stringable, EqualityComparable):
    """Represents a new slice implementation that corrects incorrect behaviour in the existing implementation.
    """

    var start: OptionalReg[Int]
    """The starting index of the slice."""
    var end: OptionalReg[Int]
    """The end index of the slice."""
    var step: Int
    """The step increment value of the slice."""

    @always_inline("nodebug")
    fn __init__(inout self, start: Int, end: Int):
        """Construct slice given the start and end values.

        Args:
            start: The start value.
            end: The end value.
        """
        self.start = start
        self.end = end
        self.step = 1

    @always_inline("nodebug")
    fn __init__(
        inout self,
        start: OptionalReg[Int],
        end: OptionalReg[Int],
        step: OptionalReg[Int],
    ):
        """Construct slice given the start, end and step values.

        Args:
            start: The start value.
            end: The end value.
            step: The step value.
        """
        self.start = start
        self.end = end

        if step and step.value() == 0:
            abort("Cannot build slice with a step of 0")

        self.step = step.value() if step else 1

    fn __str__(self) -> String:
        """Gets the string representation of the span.

        Returns:
            The string representation of the span.
        """
        var res = str(self.start.value()) if self.start else ""
        res += ":"
        if self.end:
            res += str(self.end.value()) if self.end else ""
        res += ":"
        res += str(self.step)
        return res

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Compare this slice to the other.

        Args:
            other: The slice to compare to.

        Returns:
            True if start, end, and step values of this slice match the
            corresponding values of the other slice and False otherwise.
        """
        return (
            _compare_optional(self.start, other.start)
            and _compare_optional(self.end, other.end)
            and self.step == other.step
        )

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Compare this slice to the other.

        Args:
            other: The slice to compare to.

        Returns:
            False if start, end, and step values of this slice match the
            corresponding values of the other slice and True otherwise.
        """
        return not (self == other)

    @always_inline
    fn unsafe_indices(self) -> Int:
        """Return the length of the slice.

        Only use this function if start/end is guaranteed to be not None.

        Returns:
            The length of the slice.
        """

        return len(range(self.start.value(), self.end.value(), self.step))

    fn indices(self, src_len: Int) -> (Int, Int, Int):
        """Returns a tuple of 3 intergers representing the start, end, and step
           of the slice if applied to a container of the given length.

        Uses the target container length to normalize negative, out of bounds, or None indices.

        Negative indices are wrapped using the length of the container.
        ```mojo
        s = slice_new(0, -1, 1)
        s.indices(5) # returns (0, 4, 1)
        ```

        None indices are defaulted to the start or the end of the container based
        on whether `step` is positive or negative.
        ```mojo
        s = slice_new(None, None, 1)
        s.indices(5) # returns (0, 5, 1)
        ```

        Out of bounds indices are clamped using the size of the container.
        ```mojo
        s = slice_new(20)
        s.indices(5) # returns (0, 5, 1)
        ```

        Args:
            src_len: The length of the target container.

        Returns:
            A tuple containing three integers for start, end, and step.
        """
        var start = self.start
        var end = self.end
        var positive_step = self.step > 0

        if not start:
            start = 0 if positive_step else src_len - 1
        elif start.value() < 0:
            start = start.value() + src_len
            if start.value() < 0:
                start = 0 if positive_step else -1
        elif start.value() >= src_len:
            start = src_len if positive_step else src_len - 1

        if not end:
            end = src_len if positive_step else -1
        elif end.value() < 0:
            end = end.value() + src_len
            if end.value() < 0:
                end = 0 if positive_step else -1
        elif end.value() >= src_len:
            end = src_len if positive_step else src_len - 1

        return (start.value(), end.value(), self.step)


@always_inline("nodebug")
fn slice_new(end: Int) -> SliceNew:
    """Construct slice given the end value.

    Args:
        end: The end value.

    Returns:
        The constructed slice.
    """
    return SliceNew(0, end)


@always_inline("nodebug")
fn slice_new(start: Int, end: Int) -> SliceNew:
    """Construct slice given the start and end values.

    Args:
        start: The start value.
        end: The end value.

    Returns:
        The constructed slice.
    """
    return SliceNew(start, end)


@always_inline("nodebug")
fn slice_new(
    start: OptionalReg[Int], end: OptionalReg[Int], step: OptionalReg[Int]
) -> SliceNew:
    """Construct a Slice given the start, end and step values.

    Args:
        start: The start value.
        end: The end value.
        step: The step value.

    Returns:
        The constructed slice.
    """
    return SliceNew(start, end, step)
