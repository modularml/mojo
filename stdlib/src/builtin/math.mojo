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
"""Defines basic math functions for use in the open source parts of the standard
library since the `math` package is currently closed source and cannot be
depended on in the open source parts of the standard library.

These are Mojo built-ins, so you don't need to import them.
"""

# ===----------------------------------------------------------------------=== #
# abs
# ===----------------------------------------------------------------------=== #


trait Absable:
    """
    The `Absable` trait describes a type that defines an absolute value
    operation.

    Types that conform to `Absable` will work with the builtin `abs` function.
    The absolute value operation always returns the same type as the input.

    For example:
    ```mojo
    struct Point(Absable):
        var x: Float64
        var y: Float64

        fn __abs__(self) -> Self:
            return sqrt(self.x * self.x + self.y * self.y)
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __abs__(self) -> Self:
        """Get the absolute value of this instance.

        Returns:
            The absolute value of the instance.
        """
        ...


@always_inline
fn abs[T: Absable](value: T) -> T:
    """Get the absolute value of the given object.

    Parameters:
        T: The type conforming to Absable.

    Args:
        value: The object to get the absolute value of.

    Returns:
        The absolute value of the object.
    """
    return value.__abs__()


# TODO: https://github.com/modularml/modular/issues/38694
# TODO: Remove this
@always_inline
fn abs(value: IntLiteral) -> IntLiteral:
    """Get the absolute value of the given IntLiteral.

    Args:
        value: The IntLiteral to get the absolute value of.

    Returns:
        The absolute value of the IntLiteral.
    """
    return value.__abs__()


# TODO: https://github.com/modularml/modular/issues/38694
# TODO: Remove this
@always_inline
fn abs(value: FloatLiteral) -> FloatLiteral:
    """Get the absolute value of the given FloatLiteral.

    Args:
        value: The FloatLiteral to get the absolute value of.

    Returns:
        The absolute value of the FloatLiteral.
    """
    return value.__abs__()


# ===----------------------------------------------------------------------=== #
# divmod
# ===----------------------------------------------------------------------=== #


fn divmod(numerator: Int, denominator: Int) -> Tuple[Int, Int]:
    """Performs integer division and returns the quotient and the remainder.

    Currently supported only for integers. Support for more standard library
    types like Int8, Int16... is planned.

    This method calls `a.__divmod__(b)`, thus, the actual implementation of
    divmod should go in the `__divmod__` method of the struct of `a`.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        A `Tuple` containing the quotient and the remainder.
    """
    return numerator.__divmod__(denominator)


fn divmod(numerator: UInt, denominator: UInt) -> Tuple[UInt, UInt]:
    """Performs integer division and returns the quotient and the remainder.

    Currently supported only for integers. Support for more standard library
    types like Int8, Int16... is planned.

    This method calls `a.__divmod__(b)`, thus, the actual implementation of
    divmod should go in the `__divmod__` method of the struct of `a`.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        A `Tuple` containing the quotient and the remainder.
    """
    return numerator.__divmod__(denominator)


# ===----------------------------------------------------------------------=== #
# max
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn max(x: Int, y: Int, /) -> Int:
    """Gets the maximum of two integers.

    Args:
        x: Integer input to max.
        y: Integer input to max.

    Returns:
        Maximum of x and y.
    """
    return __mlir_op.`index.maxs`(x.value, y.value)


@always_inline("nodebug")
fn max(x: UInt, y: UInt, /) -> UInt:
    """Gets the maximum of two integers.

    Args:
        x: Integer input to max.
        y: Integer input to max.

    Returns:
        Maximum of x and y.
    """
    return __mlir_op.`index.maxu`(x.value, y.value)


@always_inline("nodebug")
fn max(x: SIMD, y: __type_of(x), /) -> __type_of(x):
    """Performs elementwise maximum of x and y.

    An element of the result SIMD vector will be the maximum of the
    corresponding elements in x and y.

    Constraints:
        The type of the inputs must be numeric or boolean.

    Args:
        x: First SIMD vector.
        y: Second SIMD vector.

    Returns:
        A SIMD vector containing the elementwise maximum of x and y.
    """

    @parameter
    if x.type is DType.bool:
        return max(x.cast[DType.uint8](), y.cast[DType.uint8]()).cast[x.type]()
    else:
        constrained[
            x.type.is_numeric(), "the SIMD type must be numeric or boolean"
        ]()

        return __mlir_op.`pop.max`(x.value, y.value)


# ===----------------------------------------------------------------------=== #
# min
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn min(x: Int, y: Int, /) -> Int:
    """Gets the minimum of two integers.

    Args:
        x: Integer input to min.
        y: Integer input to min.

    Returns:
        Minimum of x and y.
    """
    return __mlir_op.`index.mins`(x.value, y.value)


@always_inline("nodebug")
fn min(x: UInt, y: UInt, /) -> UInt:
    """Gets the minimum of two integers.

    Args:
        x: Integer input to min.
        y: Integer input to min.

    Returns:
        Minimum of x and y.
    """
    return __mlir_op.`index.minu`(x.value, y.value)


@always_inline("nodebug")
fn min(x: SIMD, y: __type_of(x), /) -> __type_of(x):
    """Gets the elementwise minimum of x and y.

    An element of the result SIMD vector will be the minimum of the
    corresponding elements in x and y.

    Constraints:
        The type of the inputs must be numeric or boolean.

    Args:
        x: First SIMD vector.
        y: Second SIMD vector.

    Returns:
        A SIMD vector containing the elementwise minimum of x and y.
    """

    @parameter
    if x.type is DType.bool:
        return min(x.cast[DType.uint8](), y.cast[DType.uint8]()).cast[x.type]()
    else:
        constrained[
            x.type.is_numeric(), "the SIMD type must be numeric or boolean"
        ]()

        return __mlir_op.`pop.min`(x.value, y.value)


# ===----------------------------------------------------------------------=== #
# pow
# ===----------------------------------------------------------------------=== #


trait Powable:
    """
    The `Powable` trait describes a type that defines a power operation (i.e.
    exponentiation) with the same base and exponent types.

    Types that conform to `Powable` will work with the builtin `pow` function,
    which will return the same type as the inputs.

    For example:
    ```mojo
    @value
    struct Rational(Powable):
        var numerator: Float64
        var denominator: Float64

        fn __init__(out self, numerator: Float64, denominator: Float64):
            self.numerator = numerator
            self.denominator = denominator

        fn __pow__(self, exp: Self)  -> Self:
            var exp_value = exp.numerator / exp.denominator
            return Self(pow(self.numerator, exp_value), pow(self.denominator, exp_value))
    ```

    You can now use the ** operator to exponentiate objects
    inside generic functions:

    ```mojo
    fn exponentiate[T: Powable](base: T, exp: T) -> T:
        return base ** exp

    var base = Rational(Float64(3.0), 5.0)
    var exp = Rational(Float64(1.0), 2.0)
    var res = exponentiate(base, exp)
    ```

    ```plaintext
    raising to power
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __pow__(self, exp: Self) -> Self:
        """Return the value raised to the power of the given exponent.

        Args:
            exp: The exponent value.

        Returns:
            The value of `self` raised to the power of `exp`.
        """
        ...


fn pow[T: Powable](base: T, exp: T) -> T:
    """Computes the `base` raised to the power of the `exp`.

    Parameters:
        T: A type conforming to the `Powable` trait.

    Args:
        base: The base of the power operation.
        exp: The exponent of the power operation.

    Returns:
        The `base` raised to the power of the `exp`.
    """
    return base.__pow__(exp)


fn pow(base: SIMD, exp: Int) -> __type_of(base):
    """Computes elementwise value of a SIMD vector raised to the power of the
    given integer.

    Args:
        base: The first input argument.
        exp: The second input argument.

    Returns:
        The `base` elementwise raised raised to the power of `exp`.
    """
    return base.__pow__(exp)


# ===----------------------------------------------------------------------=== #
# round
# ===----------------------------------------------------------------------=== #


trait Roundable:
    """
    The `Roundable` trait describes a type that defines a rounding operation.

    Types that conform to `Roundable` will work with the builtin `round`
    function. The round operation always returns the same type as the input.

    For example:
    ```mojo
    @value
    struct Complex(Roundable):
        var re: Float64
        var im: Float64

        fn __round__(self) -> Self:
            return Self(round(self.re), round(self.im))

        fn __round__(self, ndigits: Int) -> Self:
            return Self(round(self.re, ndigits), round(self.im, ndigits))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __round__(self) -> Self:
        """Get a rounded value for the type.

        Returns:
            The rounded value.
        """
        ...

    fn __round__(self, ndigits: Int) -> Self:
        """Get a rounded value for the type.

        Args:
            ndigits: Number of digits after the decimal point.

        Returns:
            The rounded value.
        """
        ...


@always_inline
fn round[T: Roundable, //](number: T) -> T:
    """Get the rounded value of the given object.

    Parameters:
        T: The type conforming to Roundable.

    Args:
        number: The object to get the rounded value of.

    Returns:
        The rounded value of the object.
    """
    return number.__round__()


# TODO: remove this when conformance issue for FloatLiteral is fixed.
@always_inline
fn round(number: FloatLiteral) -> FloatLiteral:
    """Get the rounded value of the given FloatLiteral.

    Args:
        number: The FloatLiteral to get the rounded value of.

    Returns:
        The rounded value of the object.
    """
    return number.__round__()


@always_inline
fn round[T: Roundable, //](number: T, ndigits: Int) -> T:
    """Get the value of this object, rounded to a specified number of
    digits after the decimal point.

    Parameters:
        T: The type conforming to Roundable.

    Args:
        number: The object to get the rounded value of.
        ndigits: The number of digits to round to.

    Returns:
        The rounded value of the object.
    """
    return number.__round__(ndigits)


# TODO: remove this when conformance issue for FloatLiteral is fixed.
@always_inline
fn round(number: FloatLiteral, ndigits: Int) -> FloatLiteral:
    """Get the  value of this FloatLiteral, rounded to a specified number of
    digits after the decimal point.

    Args:
        number: The FloatLiteral to get the rounded value of.
        ndigits: The number of digits to round to.

    Returns:
        The rounded value of the object. Positive ndigits to the right of the decimal, negative ndigits to the left.
    """
    return number.__round__(ndigits)
