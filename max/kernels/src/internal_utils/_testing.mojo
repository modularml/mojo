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


from std.collections import Optional
from std.math import exp2

import std.testing
from std.reflection import call_location, SourceLocation
from std.testing.testing import _assert_cmp_error

from std.utils.numerics import FPUtils


# ===----------------------------------------------------------------------=== #
# Index formatting helpers
# ===----------------------------------------------------------------------=== #


def _flat_to_nd_index(flat_idx: Int, shape: List[Int]) -> String:
    """Convert a flat index to an N-dimensional index string.

    Args:
        flat_idx: The flat (linear) index into the buffer.
        shape: The shape of the N-dimensional buffer.

    Returns:
        A string representation of the N-dimensional index, e.g., "[2, 3, 4]".
    """
    if len(shape) == 0:
        return String(t"i={flat_idx}")

    # Compute N-dimensional indices from flat index (row-major order).
    # `next_digit` is called for i in [0, len(shape)), so it walks
    # dimensions from last to first -- matching the row-major unravel order.
    var remaining = flat_idx

    def next_digit(i: Int) {mut remaining, imm shape} -> Int:
        var dim_size = shape[len(shape) - 1 - i]
        var digit = remaining % dim_size
        remaining //= dim_size
        return digit

    var indices = List(length=len(shape), fill_with=next_digit)

    # Build string in correct order (indices were computed in reverse)
    var result = String("[")
    for idx in range(len(indices) - 1, -1, -1):
        if idx < len(indices) - 1:
            result += ", "
        result += String(indices[idx])
    result += "]"
    return result


def _format_index(i: Int, shape: List[Int]) -> String:
    """Format an index for error messages.

    Args:
        i: The flat index.
        shape: Optional shape for N-dimensional index display. If empty,
               displays just the flat index.

    Returns:
        A formatted string like "at [2, 3, 4]" or "at i=52".
    """
    if len(shape) > 0:
        return _flat_to_nd_index(i, shape)
    else:
        return String(t"i={i}")


def _check_span_length(
    a: Span[...], b: Span[...], location: SourceLocation
) raises:
    if len(a) != len(b):
        raise Error(
            t"Spans to not have equal lengths: {len(a)}, {len(b)}: {location}"
        )


# ===----------------------------------------------------------------------=== #
# assert_almost_equal
# ===----------------------------------------------------------------------=== #


@always_inline
def assert_almost_equal[
    dtype: DType,
    //,
](
    x: Span[Scalar[dtype], _],
    y: Span[Scalar[dtype], _],
    msg: String = "",
    *,
    shape: List[Int] = List[Int](),
    location: Optional[SourceLocation] = None,
    atol: Float64 = 1e-08,
    rtol: Float64 = 1e-05,
    equal_nan: Bool = False,
) raises:
    """Assert that two buffers are element-wise almost equal.

    Compares each element of `x` and `y` using the formula:
    `|x - y| <= atol + rtol * |y|`

    Args:
        x: Span to the first buffer.
        y: Span to the second buffer.
        msg: Optional message to include in assertion errors.
        shape: Optional shape for N-dimensional index display in error messages.
               If provided, error messages will show indices like "[2, 3, 4]"
               instead of flat indices like "i=52".
        location: Optional source location for error reporting.
        atol: Absolute tolerance (default: 1e-08).
        rtol: Relative tolerance (default: 1e-05).
        equal_nan: If True, NaN values in the same position are considered equal.

    Raises:
        Error: If any elements differ by more than the specified tolerances, or
            if the spans are unequal length.
    """
    var loc = location.or_else(call_location())
    _check_span_length(x, y, loc)

    assert_almost_equal(
        x.unsafe_ptr(),
        y.unsafe_ptr(),
        len(x),
        msg,
        shape=shape,
        location=loc,
        atol=atol,
        rtol=rtol,
        equal_nan=equal_nan,
    )


@always_inline
def assert_almost_equal[
    dtype: DType,
    //,
](
    x: UnsafePointer[Scalar[dtype], _],
    y: UnsafePointer[Scalar[dtype], _],
    num_elements: Int,
    msg: String = "",
    *,
    shape: List[Int] = List[Int](),
    location: Optional[SourceLocation] = None,
    atol: Float64 = 1e-08,
    rtol: Float64 = 1e-05,
    equal_nan: Bool = False,
) raises:
    """Assert that two buffers are element-wise almost equal.

    Compares each element of `x` and `y` using the formula:
    `|x - y| <= atol + rtol * |y|`

    Args:
        x: Pointer to the first buffer.
        y: Pointer to the second buffer.
        num_elements: Number of elements to compare.
        msg: Optional message to include in assertion errors.
        shape: Optional shape for N-dimensional index display in error messages.
               If provided, error messages will show indices like "[2, 3, 4]"
               instead of flat indices like "i=52".
        location: Optional source location for error reporting.
        atol: Absolute tolerance (default: 1e-08).
        rtol: Relative tolerance (default: 1e-05).
        equal_nan: If True, NaN values in the same position are considered equal.

    Raises:
        Error: If any elements differ by more than the specified tolerances.

    Example:
        ```mojo
        # Basic usage with flat index in errors:
        assert_almost_equal(a.data, b.data, a.num_elements())

        # With shape for better error messages:
        assert_almost_equal(
            a.data, b.data, a.num_elements(),
            shape=List[Int](2, 3, 4)
        )
        ```
    """
    for i in range(num_elements):
        std.testing.assert_almost_equal(
            x[i],
            y[i],
            msg=String(t"{msg} at {_format_index(i, shape)}"),
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
            location=location.or_else(call_location()),
        )


# ===----------------------------------------------------------------------=== #
# assert_equal
# ===----------------------------------------------------------------------=== #


@always_inline
def assert_equal[
    dtype: DType,
    //,
](
    x: UnsafePointer[Scalar[dtype], _],
    y: UnsafePointer[Scalar[dtype], _],
    num_elements: Int,
    msg: String = "",
    *,
    shape: List[Int] = List[Int](),
    location: Optional[SourceLocation] = None,
) raises:
    """Assert that two buffers are element-wise exactly equal.

    Args:
        x: Pointer to the first buffer.
        y: Pointer to the second buffer.
        num_elements: Number of elements to compare.
        msg: Optional message to include in assertion errors.
        shape: Optional shape for N-dimensional index display in error messages.
               If provided, error messages will show indices like "[2, 3, 4]"
               instead of flat indices like "i=52".
        location: Optional source location for error reporting.

    Raises:
        Error: If any elements are not exactly equal.

    Example:
        ```mojo
        # Basic usage:
        assert_equal(a.data, b.data, a.num_elements())

        # With shape for better error messages:
        assert_equal(
            a.data, b.data, a.num_elements(),
            shape=List[Int](2, 3, 4)
        )
        ```
    """
    for i in range(num_elements):
        std.testing.assert_equal(
            x[i],
            y[i],
            msg=String(t"{msg} at {_format_index(i, shape)}"),
            location=location.or_else(call_location()),
        )


@always_inline
def assert_equal[
    dtype: DType,
    //,
](
    x: Span[Scalar[dtype], _],
    y: Span[Scalar[dtype], _],
    msg: String = "",
    *,
    shape: List[Int] = List[Int](),
    location: Optional[SourceLocation] = None,
) raises:
    """Assert that two spans are element-wise exactly equal.

    Args:
        x: Span of the first buffer.
        y: Span of the second buffer.
        msg: Optional message to include in assertion errors.
        shape: Optional shape for N-dimensional index display in error messages.
                If provided, error messages will show indices like "[2, 3, 4]"
                instead of flat indices like "i=52".
        location: Optional source location for error reporting.

    Raises:
        Error: If any elements are not exactly equal or if the spans have unequal
            length.
    """
    var loc = location.or_else(call_location())
    _check_span_length(x, y, loc)
    assert_equal(
        x.unsafe_ptr(), y.unsafe_ptr(), len(x), msg, shape=shape, location=loc
    )


# ===----------------------------------------------------------------------=== #
# assert_with_measure
# ===----------------------------------------------------------------------=== #


@always_inline
def assert_with_measure[
    dtype: DType,
    //,
    measure: def[dtype: DType](
        UnsafePointer[mut=False, Scalar[dtype], _],
        UnsafePointer[mut=False, Scalar[dtype], _],
        Int,
    ) thin -> Float64,
](
    x: UnsafePointer[Scalar[dtype], _],
    y: UnsafePointer[Scalar[dtype], _],
    num_elements: Int,
    msg: String = "",
    *,
    location: Optional[SourceLocation] = None,
    threshold: Optional[Float64] = None,
) raises:
    """Assert that a custom measure between two buffers is below a threshold.

    Computes a measure (e.g., correlation, KL divergence) between `x` and `y`,
    and asserts that it does not exceed the specified threshold.

    Args:
        x: Pointer to the first buffer.
        y: Pointer to the second buffer.
        num_elements: Number of elements in each buffer.
        msg: Optional message to include in assertion errors.
        location: Optional source location for error reporting.
        threshold: Maximum allowed value for the measure. If not specified,
                   defaults to sqrt(machine epsilon) for the dtype.

    Parameters:
        dtype: The data type of the buffer elements.
        measure: A function that computes a scalar measure between two buffers.
                 Signature: `def[dtype](ptr1, ptr2, n) -> Float64`

    Raises:
        Error: If the computed measure exceeds the threshold.

    Example:
        ```mojo
        from internal_utils._measure import relative_difference

        assert_with_measure[relative_difference](
            a.data, b.data, a.num_elements(),
            threshold=0.001
        )
        ```
    """
    comptime sqrt_eps = exp2(
        -0.5 * Float64(FPUtils[dtype].mantissa_width())
    ).cast[.float64]()
    var m = measure(
        x.address_space_cast[.GENERIC](),
        y.address_space_cast[.GENERIC](),
        num_elements,
    )
    var t = threshold.or_else(sqrt_eps)
    if m > t:
        raise _assert_cmp_error["`left > right`, left = measure"](
            String(m),
            String(t),
            msg=msg,
            loc=location.or_else(call_location()),
        )


# ===----------------------------------------------------------------------=== #
# pytorch_like_tolerances_for
# ===----------------------------------------------------------------------=== #


@always_inline
def pytorch_like_tolerances_for[dtype: DType]() -> Tuple[Float64, Float64]:
    """Get PyTorch-like default tolerances for a given dtype.

    Returns tolerance values modeled after PyTorch's default tolerances
    for floating-point comparisons.

    Parameters:
        dtype: The data type to get tolerances for.

    Returns:
        A tuple of (rtol, atol) - relative and absolute tolerances.

    Example:
        ```mojo
        rtol, atol = pytorch_like_tolerances_for[.float16]()
        assert_almost_equal(x, y, n, rtol=rtol, atol=atol)
        ```
    """

    comptime if dtype == .float16:
        return (1e-3, 1e-5)
    elif dtype == .bfloat16:
        return (1.6e-2, 1e-5)
    elif dtype == .float32:
        return (1.3e-6, 1e-5)
    elif dtype == .float64:
        return (1e-7, 1e-7)
    else:
        return (0.0, 0.0)


# ===----------------------------------------------------------------------=== #
# test_value_for_gpu_element
# ===----------------------------------------------------------------------=== #


@always_inline
@__parameter
def test_value_for_gpu_element[
    dtype: DType,
    modulo: Int = 251 if dtype == .float32 else 13,
](gpu_rank: Int, element_idx: Int) -> Scalar[dtype]:
    """Generates unique deterministic test values per GPU and element index.

    Creates predictable values for testing multi-GPU operations where each
    GPU's contribution needs to be distinguishable. Uses prime modulus to
    avoid power-of-two aliasing patterns. 251 is the largest prime < 256.

    Args:
        gpu_rank: The rank/ID of the GPU (0-indexed).
        element_idx: The element index within the buffer.

    Returns:
        A unique scalar value for this GPU and element combination.

    Examples:
        `test_value_for_gpu_element[.float32](0, 0)` !=
        `test_value_for_gpu_element[.float32](1, 0)`.
    """
    return Scalar[dtype](gpu_rank + 1) + Scalar[dtype](element_idx % modulo)
