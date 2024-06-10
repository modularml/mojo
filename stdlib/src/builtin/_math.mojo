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
# gcd
# ===----------------------------------------------------------------------=== #


fn gcd(owned m: Int, owned n: Int, /) -> Int:
    """Compute the greatest common divisor of two integers.

    Args:
        m: The first integer.
        n: The second integrer.

    Returns:
        The greatest common divisor of the two integers.
    """
    while n:
        m, n = n, m % n
    return abs(m)


fn gcd(s: Span[Int], /) -> Int:
    """Computes the greatest common divisor of a span of integers.

    Args:
        s: A span containing a collection of integers.

    Returns:
        The greatest common divisor of all the integers in the span.
    """
    if len(s) == 0:
        return 0
    var result = s[0]
    for item in s[1:]:
        result = gcd(item[], result)
        if result == 1:
            return result
    return result


@always_inline
fn gcd(l: List[Int], /) -> Int:
    """Computes the greatest common divisor of a list of integers.

    Args:
        l: A list containing a collection of integers.

    Returns:
        The greatest common divisor of all the integers in the list.
    """
    return gcd(Span(l))


fn gcd(*values: Int) -> Int:
    """Computes the greatest common divisor of a variadic number of integers.

    Args:
        values: A variadic list of integers.

    Returns:
        The greatest common divisor of the given integers.
    """
    # TODO: Deduplicate when we can create a Span from VariadicList
    if len(values) == 0:
        return 0
    var result = values[0]
    for i in range(1, len(values)):
        result = gcd(values[i], result)
        if result == 1:
            return result
    return result


# ===----------------------------------------------------------------------=== #
# lcm
# ===----------------------------------------------------------------------=== #


fn lcm(owned m: Int, owned n: Int, /) -> Int:
    """Computes the least common multiple of two integers.

    Args:
        m: The first integer.
        n: The second integer.

    Returns:
        The least common multiple of the two integers.
    """
    var d: Int
    if d := gcd(m, n):
        return abs((m // d) * n if m > n else (n // d) * m)
    return 0


fn lcm(s: Span[Int], /) -> Int:
    """Computes the least common multiple of a span of integers.

    Args:
        s: A span of integers.

    Returns:
        The least common multiple of the span.
    """
    if len(s) == 0:
        return 1

    var result = s[0]
    for item in s[1:]:
        result = lcm(result, item[])
    return result


@always_inline
fn lcm(l: List[Int], /) -> Int:
    """Computes the least common multiple of a list of integers.

    Args:
        l: A list of integers.

    Returns:
        The least common multiple of the list.
    """
    return lcm(Span(l))


fn lcm(*values: Int) -> Int:
    """Computes the least common multiple of a variadic list of integers.

    Args:
        values: A variadic list of integers.

    Returns:
        The least common multiple of the list.
    """
    # TODO: Deduplicate when we can create a Span from VariadicList
    if len(values) == 0:
        return 1

    var result = values[0]
    for i in range(1, len(values)):
        result = lcm(result, values[i])
    return result
