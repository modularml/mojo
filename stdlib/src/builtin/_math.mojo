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
"""Module to contain some components of the future math module.

This is needed to work around some circular dependencies; all elements of this
module should be exposed by the current `math` module. The contents of this
module should be eventually moved to the `math` module when it's open sourced.
"""

from bit import countr_zero

# ===----------------------------------------------------------------------=== #
# Ceilable
# ===----------------------------------------------------------------------=== #


trait Ceilable:
    """
    The `Ceilable` trait describes a type that defines a ceiling operation.

    Types that conform to `Ceilable` will work with the builtin `ceil`
    function. The ceiling operation always returns the same type as the input.

    For example:
    ```mojo
    from math import Ceilable, ceil

    @value
    struct Complex(Ceilable):
        var re: Float64
        var im: Float64

        fn __ceil__(self) -> Self:
            return Self(ceil(re), ceil(im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __ceil__(self) -> Self:
        ...


# ===----------------------------------------------------------------------=== #
# Floorable
# ===----------------------------------------------------------------------=== #


trait Floorable:
    """
    The `Floorable` trait describes a type that defines a floor operation.

    Types that conform to `Floorable` will work with the builtin `floor`
    function. The floor operation always returns the same type as the input.

    For example:
    ```mojo
    from math import Floorable, floor

    @value
    struct Complex(Floorable):
        var re: Float64
        var im: Float64

        fn __floor__(self) -> Self:
            return Self(floor(re), floor(im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __floor__(self) -> Self:
        ...


# ===----------------------------------------------------------------------=== #
# CeilDivable
# ===----------------------------------------------------------------------=== #


trait CeilDivable:
    """
    The `CeilDivable` trait describes a type that defines a ceil division
    operation.

    Types that conform to `CeilDivable` will work with the `math.ceildiv`
    function.

    For example:
    ```mojo
    from math import CeilDivable

    @value
    struct Foo(CeilDivable):
        var x: Float64

        fn __floordiv__(self, other: Self) -> Self:
            return self.x // other.x

        fn __rfloordiv__(self, other: Self) -> Self:
            return other // self

        fn __neg__(self) -> Self:
            return -self.x
    ```
    """

    # TODO(MOCO-333): Reconsider these signatures when we have parametric traits
    # or associated types.
    fn __floordiv__(self, other: Self) -> Self:
        ...

    fn __rfloordiv__(self, other: Self) -> Self:
        ...

    fn __neg__(self) -> Self:
        ...


trait CeilDivableRaising:
    """
    The `CeilDivable` trait describes a type that define a floor division and
    negation operation that can raise.

    Types that conform to `CeilDivableRaising` will work with the `//` operator
    as well as the `math.ceildiv` function.

    For example:
    ```mojo
    from math import CeilDivableRaising

    @value
    struct Foo(CeilDivableRaising):
        var x: object

        fn __floordiv__(self, other: Self) raises -> Self:
            return self.x // other.x

        fn __rfloordiv__(self, other: Self) raises -> Self:
            return other // self

        fn __neg__(self) raises -> Self:
            return -self.x
    ```
    """

    # TODO(MOCO-333): Reconsider these signatures when we have parametric traits
    # or associated types.
    fn __floordiv__(self, other: Self) raises -> Self:
        ...

    fn __rfloordiv__(self, other: Self) raises -> Self:
        ...

    fn __neg__(self) raises -> Self:
        ...


# ===----------------------------------------------------------------------=== #
# Truncable
# ===----------------------------------------------------------------------=== #


trait Truncable:
    """
    The `Truncable` trait describes a type that defines a truncation operation.

    Types that conform to `Truncable` will work with the builtin `trunc`
    function. The truncation operation always returns the same type as the
    input.

    For example:
    ```mojo
    from math import Truncable, trunc

    @value
    struct Complex(Truncable):
        var re: Float64
        var im: Float64

        fn __trunc__(self) -> Self:
            return Self(trunc(re), trunc(im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __trunc__(self) -> Self:
        ...


# ===----------------------------------------------------------------------=== #
# clamp
# ===----------------------------------------------------------------------=== #


fn clamp(value: Int, lower: Int, upper: Int) -> Int:
    """Clamp value to the range of [lower, upper].

    If value is less lower, or greater than upper, it becomes the respective
    value, else value is returned.

    If lower > upper, then the two are swapped.

    Args:
        value: The integer to clamp.
        lower: The lower bound of the range.
        upper: Then upper bound of the range.

    Returns:
        A value in the range [lower, upper].
    """
    var lo = lower
    var hi = upper
    if lo > hi:
        swap(lo, hi)
    return max(lo, min(value, hi))


fn clamp[
    type: DType, size: Int, //
](value: SIMD[type, size], lower: Scalar[type], upper: Scalar[type]) -> SIMD[
    type, size
]:
    """Clamp values to the range of [lower, upper].

    If value is less lower, or greater than upper, it becomes the respective
    value, else value is returned.

    If value is NAN, then upper is returned.

    If lower > upper, then the two are swapped.

    This is equivalent to `value.clamp(lower, upper)`

    Parameters:
        type: The dtype of the arguments.
        size: The SIMD size of the arguments.

    Args:
        value: The value to clamp.
        lower: The lower bound of the range.
        upper: Then upper bound of the range.

    Returns:
        Values in the range [lower, upper].
    """
    var lo = lower
    var hi = upper
    if lo > hi:
        swap(lo, hi)
    return value.clamp(lo, hi)
