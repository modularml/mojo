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
"""Implements slice.

These are Mojo built-ins, so you don't need to import them.
"""

from std.format._utils import FormatStruct, Named


struct Slice(
    Equatable,
    ImplicitlyCopyable,
    Writable,
):
    """Represents a slice expression.

    Objects of this type are generated when slice syntax is used within square
    brackets, e.g.:

    ```mojo
    var lst: List[Int] = [0,1,2,3,4,5,6,7]

    # Both are equivalent and result in a list: [].
    var l1 = List(lst[6:])
    var l2 = lst.__getitem__(Slice(6, len(lst)))
    ```
    """

    # Fields
    var start: Optional[Int]
    """The starting index of the slice."""
    var end: Optional[Int]
    """The end index of the slice."""
    var step: Optional[Int]
    """The step increment value of the slice."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __init__(out self, start: Int, end: Int):
        """Construct slice given the start and end values.

        Args:
            start: The start value.
            end: The end value.
        """
        self.start = start
        self.end = end
        self.step = None

    @always_inline
    def __init__(
        out self,
        start: Optional[Int],
        end: Optional[Int],
        step: Optional[Int],
        __slice_literal__: NoneType = None,
    ):
        """Construct slice given the start, end and step values.

        Args:
            start: The start value.
            end: The end value.
            step: The step value.
            __slice_literal__: Enables slice literal syntax.
        """
        self.start = start
        self.end = end
        self.step = step

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Write Slice string representation to a `Writer`.

        Args:
            writer: The object to write to.
        """
        FormatStruct(writer, "Slice").fields(self.start, self.end, self.step)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write Slice string representation to a `Writer`.

        Args:
            writer: The object to write to.
        """
        FormatStruct(writer, "Slice").fields(
            Named("start", self.start),
            Named("end", self.end),
            Named("step", self.step),
        )

    @always_inline
    def __eq__(self, other: Self) -> Bool:
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

    def indices(self, length: Int) -> Tuple[Int, Int, Int]:
        """Returns a tuple of 3 integers representing the start, end, and step
           of the slice if applied to a container of the given length.

        Uses the target container length to normalize negative, out of bounds,
        or None indices.

        Negative indices are wrapped using the length of the container.
        ```mojo
        s = slice(0, -1, 1)
        i = s.indices(5) # returns (0, 4, 1)
        ```

        None indices are defaulted to the start or the end of the container
        based on whether `step` is positive or negative.
        ```mojo
        s = slice(None, None, 1)
        i = s.indices(5) # returns (0, 5, 1)
        ```

        Out of bounds indices are clamped using the size of the container.
        ```mojo
        s = slice(20)
        i = s.indices(5) # returns (0, 5, 1)
        ```

        Args:
            length: The length of the target container.

        Returns:
            A tuple containing three integers for start, end, and step.
        """
        var step = self.step.or_else(1)

        var start = self.start
        var end = self.end

        var positive_step = step > 0

        if not start:
            start = 0 if positive_step else length - 1
        elif start.value() < 0:
            start = start.value() + length
            if start.value() < 0:
                start = 0 if positive_step else -1
        elif start.value() >= length:
            start = length if positive_step else length - 1

        if not end:
            end = length if positive_step else -1
        elif end.value() < 0:
            end = end.value() + length
            if end.value() < 0:
                end = 0 if positive_step else -1
        elif end.value() >= length:
            end = length if positive_step else length - 1

        return (start.value(), end.value(), step)


struct StridedSlice(ImplicitlyCopyable, Writable):
    """Represents a slice expression that has a stride.

    This type is used to support different behavior for strided vs unstrided
    slicing.
    """

    var _inner: Slice

    @implicit
    def __init__(out self, other: Slice):
        """Implicitly convert from a general slice.

        Args:
            other: The other slice.
        """
        self._inner = other

    def __init__(
        out self,
        start: Optional[Int],
        end: Optional[Int],
        stride: Int,
        __slice_literal__: NoneType = None,
    ):
        """Construct slice given start, end, and stride values.

        Args:
            start: The start value.
            end: The end value.
            stride: The step value.
            __slice_literal__: Enables slice literal syntax.
        """

        self._inner = Slice(start, end, stride)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Write StridedSlice string representation to a `Writer`.

        Args:
            writer: The object to write to.
        """
        self._inner.write_to(writer)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write StridedSlice debug representation to a `Writer`.

        Args:
            writer: The object to write to.
        """
        self._inner.write_repr_to(writer)

    def indices(self, length: Int) -> Tuple[Int, Int, Int]:
        """Returns a tuple of 3 integers representing start, end, and step
        of the slice if applied to a container of given length.

        Args:
            length: The length of the target container.

        Returns:
            A tuple containing three integers for start, end, and step.
        """
        return self._inner.indices(length)


struct ContiguousSlice(ImplicitlyCopyable, Writable):
    """Represents a slice expression without a stride.

    This type is used to support different behavior for strided vs unstrided
    slicing.
    """

    var start: Optional[Int]
    """The starting index of the slice."""
    var end: Optional[Int]
    """The end index of the slice."""

    @always_inline
    def __init__(
        out self,
        start: Optional[Int],
        end: Optional[Int],
        stride: NoneType,
        __slice_literal__: NoneType = None,
    ):
        """Construct slice given the start and end values.

        Args:
            start: The start value.
            end: The end value.
            stride: Always none. Disambiguates from slices with a stride.
            __slice_literal__: Enables slice literal syntax.
        """
        self.start = start
        self.end = end

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Write ContiguousSlice string representation to a `Writer`.

        Args:
            writer: The object to write to.
        """
        Slice(self.start, self.end, None).write_to(writer)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write ContiguousSlice debug representation to a `Writer`.

        Args:
            writer: The object to write to.
        """
        Slice(self.start, self.end, None).write_repr_to(writer)

    def indices(self, length: Int) -> Tuple[Int, Int]:
        """Returns a tuple of 2 integers representing the start, and end
        of the slice if applied to a container of the given length.

        Args:
            length: The length of the target container.

        Returns:
            A tuple containing two integers for start and end.
        """
        var start = self.start.or_else(0)
        var end = self.end.or_else(length)
        if start < 0:
            start = max(start + length, 0)
        elif start >= length:
            start = length

        if end < 0:
            end = max(end + length, 0)
        elif end >= length:
            end = length

        return start, end


# ===-----------------------------------------------------------------------===#
# Slice constructor functions
# ===-----------------------------------------------------------------------===#


@always_inline
def slice(end: Int) -> Slice:
    """Construct slice given the end value.

    Args:
        end: The end value.

    Returns:
        The constructed slice.
    """
    return Slice(None, end, None)


@always_inline
def slice(start: Int, end: Int) -> Slice:
    """Construct slice given the start and end values.

    Args:
        start: The start value.
        end: The end value.

    Returns:
        The constructed slice.
    """
    return Slice(start, end)


@always_inline
def slice(
    start: Optional[Int], end: Optional[Int], step: Optional[Int]
) -> Slice:
    """Construct a Slice given the start, end and step values.

    Args:
        start: The start value.
        end: The end value.
        step: The step value.

    Returns:
        The constructed slice.
    """
    return Slice(start, end, step)
