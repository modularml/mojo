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
"""Defines math utilities.

You can import these APIs from the `math` package. For example:

```mojo
from math import floor
```
"""

from collections import List
from sys import (
    bitwidthof,
    has_avx512f,
    is_amd_gpu,
    is_nvidia_gpu,
    llvm_intrinsic,
    simdwidthof,
    sizeof,
)
from sys._assembly import inlined_assembly
from sys.ffi import _external_call_const
from sys.info import _current_arch

from bit import count_trailing_zeros
from builtin.dtype import _integral_type_of
from builtin.simd import _modf, _simd_apply
from memory import UnsafePointer, Span

from utils.index import IndexList
from utils.numerics import FPUtils, isnan, nan
from utils.static_tuple import StaticTuple

from .polynomial import polynomial_evaluate

# ===----------------------------------------------------------------------=== #
# floor
# ===----------------------------------------------------------------------=== #


@always_inline
fn floor[T: Floorable, //](value: T) -> T:
    """Get the floor value of the given object.

    Parameters:
        T: The type conforming to Floorable.

    Args:
        value: The object to get the floor value of.

    Returns:
        The floor value of the object.
    """
    return value.__floor__()


# ===----------------------------------------------------------------------=== #
# ceil
# ===----------------------------------------------------------------------=== #


@always_inline
fn ceil[T: Ceilable, //](value: T) -> T:
    """Get the ceiling value of the given object.

    Parameters:
        T: The type conforming to Ceilable.

    Args:
        value: The object to get the ceiling value of.

    Returns:
        The ceiling value of the object.
    """
    return value.__ceil__()


# ===----------------------------------------------------------------------=== #
# ceildiv
# ===----------------------------------------------------------------------=== #


@always_inline
fn ceildiv[T: CeilDivable, //](numerator: T, denominator: T) -> T:
    """Return the rounded-up result of dividing numerator by denominator.

    Parameters:
        T: A type that support floor division.

    Args:
        numerator: The numerator.
        denominator: The denominator.

    Returns:
        The ceiling of dividing numerator by denominator.
    """
    # return -(numerator // -denominator)
    return numerator.__ceildiv__(denominator)


@always_inline
fn ceildiv[T: CeilDivableRaising, //](numerator: T, denominator: T) raises -> T:
    """Return the rounded-up result of dividing numerator by denominator, potentially raising.

    Parameters:
        T: A type that support floor division.

    Args:
        numerator: The numerator.
        denominator: The denominator.

    Returns:
        The ceiling of dividing numerator by denominator.
    """
    return numerator.__ceildiv__(denominator)


# NOTE: this overload is needed because of overload precedence; without it the
# Int overload would be preferred, and ceildiv wouldn't work on IntLiteral.
@always_inline
fn ceildiv(numerator: IntLiteral, denominator: IntLiteral) -> IntLiteral:
    """Return the rounded-up result of dividing numerator by denominator.

    Args:
        numerator: The numerator.
        denominator: The denominator.

    Returns:
        The ceiling of dividing numerator by denominator.
    """
    return numerator.__ceildiv__(denominator)


# ===----------------------------------------------------------------------=== #
# trunc
# ===----------------------------------------------------------------------=== #


@always_inline
fn trunc[T: Truncable, //](value: T) -> T:
    """Get the truncated value of the given object.

    Parameters:
        T: The type conforming to Truncable.

    Args:
        value: The object to get the truncated value of.

    Returns:
        The truncated value of the object.
    """
    return value.__trunc__()


# ===----------------------------------------------------------------------=== #
# sqrt
# ===----------------------------------------------------------------------=== #


@always_inline
fn sqrt(x: Int) -> Int:
    """Performs square root on an integer.

    Args:
        x: The integer value to perform square root on.

    Returns:
        The square root of x.
    """
    if x < 0:
        return 0

    var r = 0
    var r2 = 0

    @parameter
    for p in range(bitwidthof[Int]() // 4, -1, -1):
        var tr2 = r2 + (r << (p + 1)) + (1 << (p + p))
        if tr2 <= x:
            r2 = tr2
            r |= 1 << p

    return r


@always_inline
fn _sqrt_nvvm(x: SIMD) -> __type_of(x):
    constrained[
        x.type in (DType.float32, DType.float64), "must be f32 or f64 type"
    ]()
    alias instruction = "llvm.nvvm.sqrt.approx.ftz.f" if x.type is DType.float32 else "llvm.nvvm.sqrt.approx.d"
    var res = __type_of(x)()

    @parameter
    for i in range(x.size):
        res[i] = llvm_intrinsic[
            instruction, Scalar[x.type], has_side_effect=False
        ](x[i])
    return res


@always_inline
fn sqrt[
    type: DType, simd_width: Int, //
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Performs elementwise square root on the elements of a SIMD vector.

    Constraints:
        The type must be an arithmetic or boolean type.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector to perform square root on.

    Returns:
        The elementwise square root of x.
    """
    constrained[
        type.is_numeric() or type is DType.bool,
        "type must be arithmetic or boolean",
    ]()

    @parameter
    if type is DType.bool:
        return x
    elif type.is_integral():
        var res = SIMD[type, simd_width]()

        @parameter
        for i in range(simd_width):
            res[i] = sqrt(int(x[i]))
        return res
    elif is_nvidia_gpu():

        @parameter
        if x.type in (DType.float16, DType.bfloat16):
            return _sqrt_nvvm(x.cast[DType.float32]()).cast[x.type]()
        return _sqrt_nvvm(x)

    return llvm_intrinsic["llvm.sqrt", __type_of(x), has_side_effect=False](x)


# ===----------------------------------------------------------------------=== #
# rsqrt
# ===----------------------------------------------------------------------=== #


@always_inline
fn _isqrt_nvvm(x: SIMD) -> __type_of(x):
    constrained[
        x.type in (DType.float32, DType.float64), "must be f32 or f64 type"
    ]()

    alias instruction = "llvm.nvvm.rsqrt.approx.ftz.f" if x.type is DType.float32 else "llvm.nvvm.rsqrt.approx.d"
    var res = __type_of(x)()

    @parameter
    for i in range(x.size):
        res[i] = llvm_intrinsic[
            instruction, Scalar[x.type], has_side_effect=False
        ](x[i])
    return res


@always_inline
fn isqrt(x: SIMD) -> __type_of(x):
    """Performs elementwise reciprocal square root on a SIMD vector.

    Args:
        x: SIMD vector to perform reciprocal square root on.

    Returns:
        The elementwise reciprocal square root of x.
    """
    constrained[x.type.is_floating_point(), "type must be floating point"]()

    @parameter
    if is_nvidia_gpu():

        @parameter
        if x.type in (DType.float16, DType.bfloat16):
            return _isqrt_nvvm(x.cast[DType.float32]()).cast[x.type]()

        return _isqrt_nvvm(x)

    return 1 / sqrt(x)


# ===----------------------------------------------------------------------=== #
# recip
# ===----------------------------------------------------------------------=== #


@always_inline
fn _recip_nvvm(x: SIMD) -> __type_of(x):
    constrained[
        x.type in (DType.float32, DType.float64), "must be f32 or f64 type"
    ]()

    alias instruction = "llvm.nvvm.rcp.approx.ftz.f" if x.type is DType.float32 else "llvm.nvvm.rcp.approx.ftz.d"
    var res = __type_of(x)()

    @parameter
    for i in range(x.size):
        res[i] = llvm_intrinsic[
            instruction, Scalar[x.type], has_side_effect=False
        ](x[i])
    return res


@always_inline
fn recip(x: SIMD) -> __type_of(x):
    """Performs elementwise reciprocal on a SIMD vector.

    Args:
        x: SIMD vector to perform reciprocal on.

    Returns:
        The elementwise reciprocal of x.
    """
    constrained[x.type.is_floating_point(), "type must be floating point"]()

    @parameter
    if is_nvidia_gpu():

        @parameter
        if x.type in (DType.float16, DType.bfloat16):
            return _recip_nvvm(x.cast[DType.float32]()).cast[x.type]()

        return _recip_nvvm(x)

    return 1 / x


# ===----------------------------------------------------------------------=== #
# exp2
# ===----------------------------------------------------------------------=== #


@always_inline
fn exp2[
    type: DType, simd_width: Int, //
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Computes elementwise 2 raised to the power of n, where n is an element
    of the input SIMD vector.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector to perform exp2 on.

    Returns:
        Vector containing $2^n$ computed elementwise, where n is an element in
        the input SIMD vector.
    """

    @parameter
    if is_nvidia_gpu():

        @parameter
        if type is DType.float16:

            @parameter
            if String(_current_arch()) == "sm_90a":
                return _call_ptx_intrinsic[
                    scalar_instruction="ex2.approx.f16",
                    vector2_instruction="ex2.approx.f16x2",
                    scalar_constraints="=h,h",
                    vector_constraints="=r,r",
                ](x)
            else:
                return _call_ptx_intrinsic[
                    instruction="ex2.approx.f16", constraints="=h,h"
                ](x)
        elif type is DType.bfloat16 and String(_current_arch()) == "sm_90a":
            return _call_ptx_intrinsic[
                scalar_instruction="ex2.approx.ftz.bf16",
                vector2_instruction="ex2.approx.ftz.bf16x2",
                scalar_constraints="=h,h",
                vector_constraints="=r,r",
            ](x)
        elif type is DType.float32:
            return _call_ptx_intrinsic[
                instruction="ex2.approx.ftz.f32", constraints="=f,f"
            ](x)

    @parameter
    if type not in (DType.float32, DType.float64):
        return exp2(x.cast[DType.float32]()).cast[type]()

    var xc = x.clamp(-126, 126)

    var m = xc.cast[__type_of(x.to_bits()).type]()

    xc -= m.cast[type]()

    var r = polynomial_evaluate[
        List[SIMD[type, simd_width]](
            1.0,
            0.693144857883,
            0.2401793301105,
            5.551834031939e-2,
            9.810352697968e-3,
            1.33336498402e-3,
        ),
    ](xc)
    return __type_of(r)(
        from_bits=(r.to_bits() + (m << FPUtils[type].mantissa_width()))
    )


# ===----------------------------------------------------------------------=== #
# ldexp
# ===----------------------------------------------------------------------=== #


@always_inline
fn _ldexp_impl[
    type: DType, simd_width: Int, //
](x: SIMD[type, simd_width], exp: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Computes elementwise ldexp function.

    The ldexp function multiplies a floating point value x by the number 2
    raised to the exp power. I.e. $ldexp(x,exp)$ calculate the value of $x *
    2^{exp}$ and is used within the $erf$ function.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector of floating point values.
        exp: SIMD vector containing the exponents.

    Returns:
        Vector containing elementwise result of ldexp on x and exp.
    """

    alias hardware_simd_width = simdwidthof[type]()

    @parameter
    if (
        has_avx512f()
        and type is DType.float32
        and simd_width >= hardware_simd_width
    ):
        var res: SIMD[type, simd_width] = 0
        var zero: SIMD[type, hardware_simd_width] = 0

        @parameter
        for idx in range(simd_width // hardware_simd_width):
            alias i = idx * hardware_simd_width
            # On AVX512, we can use the scalef intrinsic to compute the ldexp
            # function.
            var part = llvm_intrinsic[
                "llvm.x86.avx512.mask.scalef.ps.512",
                SIMD[type, hardware_simd_width],
                has_side_effect=False,
            ](
                x.slice[hardware_simd_width, offset=i](),
                exp.slice[hardware_simd_width, offset=i](),
                zero,
                Int16(-1),
                Int32(4),
            )
            res = res.insert[offset=i](part)

        return res

    alias integral_type = FPUtils[type].integral_type
    var m = exp.cast[integral_type]() + FPUtils[type].exponent_bias()

    return x * __type_of(x)(from_bits=m << FPUtils[type].mantissa_width())


@always_inline
fn ldexp[
    type: DType, simd_width: Int, //
](x: SIMD[type, simd_width], exp: SIMD[DType.int32, simd_width]) -> SIMD[
    type, simd_width
]:
    """Computes elementwise ldexp function.

    The ldexp function multiplies a floating point value x by the number 2
    raised to the exp power. I.e. $ldexp(x,exp)$ calculate the value of $x *
    2^{exp}$ and is used within the $erf$ function.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector of floating point values.
        exp: SIMD vector containing the exponents.

    Returns:
        Vector containing elementwise result of ldexp on x and exp.
    """
    return _ldexp_impl(x, exp.cast[type]())


# ===----------------------------------------------------------------------=== #
# exp
# ===----------------------------------------------------------------------=== #


@always_inline
fn _exp_taylor[
    type: DType, simd_width: Int, //
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    alias coefficients = List[SIMD[type, simd_width]](
        1.0,
        1.0,
        0.5,
        0.16666666666666666667,
        0.041666666666666666667,
        0.0083333333333333333333,
        0.0013888888888888888889,
        0.00019841269841269841270,
        0.000024801587301587301587,
        2.7557319223985890653e-6,
        2.7557319223985890653e-7,
        2.5052108385441718775e-8,
        2.0876756987868098979e-9,
    )
    return polynomial_evaluate[
        coefficients if type is DType.float64 else coefficients[:8],
    ](x)


@always_inline
fn exp[
    type: DType, simd_width: Int, //
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Calculates elementwise exponential of the input vector.

    Given an input vector $X$ and an output vector $Y$, sets $Y_i = e^{X_i}$ for
    each position $i$ in the input vector (where $e$ is the mathematical constant
    $e$).

    Constraints:
        The input must be a floating-point type.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: The input SIMD vector.

    Returns:
        A SIMD vector containing $e$ raised to the power $X_i$ where $X_i$ is an
        element in the input SIMD vector.
    """
    constrained[type.is_floating_point(), "must be a floating point value"]()
    alias neg_ln2 = -0.69314718055966295651160180568695068359375
    alias inv_lg2 = 1.442695040888963407359924681001892137426646

    @parameter
    if is_nvidia_gpu():

        @parameter
        if type in (DType.float16, DType.float32):
            return exp2(x * inv_lg2)

    @parameter
    if type not in (DType.float32, DType.float64):
        return exp(x.cast[DType.float32]()).cast[type]()

    var min_val: SIMD[type, simd_width]
    var max_val: SIMD[type, simd_width]

    @parameter
    if type is DType.float64:
        min_val = -709.436139303
        max_val = 709.437
    else:
        min_val = -88.3762626647949
        max_val = 88.3762626647950

    var xc = x.clamp(min_val, max_val)
    var k = floor(xc.fma(inv_lg2, 0.5))
    var r = k.fma(neg_ln2, xc)
    return max(_ldexp_impl(_exp_taylor(r), k), xc)


# ===----------------------------------------------------------------------=== #
# frexp
# ===----------------------------------------------------------------------=== #


@always_inline
fn _frexp_mask1[
    simd_width: Int, type: DType
]() -> SIMD[_integral_type_of[type](), simd_width]:
    @parameter
    if type is DType.float16:
        return 0x7C00
    elif type is DType.bfloat16:
        return 0x7F80
    elif type is DType.float32:
        return 0x7F800000
    else:
        constrained[type is DType.float64, "unhandled fp type"]()
        return 0x7FF0000000000000


@always_inline
fn _frexp_mask2[
    simd_width: Int, type: DType
]() -> SIMD[_integral_type_of[type](), simd_width]:
    @parameter
    if type is DType.float16:
        return 0x3800
    elif type is DType.bfloat16:
        return 0x3F00
    elif type is DType.float32:
        return 0x3F000000
    else:
        constrained[type is DType.float64, "unhandled fp type"]()
        return 0x3FE0000000000000


@always_inline
fn frexp[
    type: DType, simd_width: Int, //
](x: SIMD[type, simd_width]) -> StaticTuple[SIMD[type, simd_width], 2]:
    """Breaks floating point values into a fractional part and an exponent part.
    This follows C and Python in increasing the exponent by 1 and normalizing the
    fraction from 0.5 to 1.0 instead of 1.0 to 2.0.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: The input values.

    Returns:
        A tuple of two SIMD vectors containing the fractional and exponent parts
        of the input floating point values.
    """
    # Based on the implementation in boost/simd/arch/common/simd/function/ifrexp.hpp
    constrained[type.is_floating_point(), "must be a floating point value"]()
    alias T = SIMD[type, simd_width]
    alias zero = T(0)
    # Add one to the resulting exponent up by subtracting 1 from the bias
    alias exponent_bias = FPUtils[type].exponent_bias() - 1
    alias mantissa_width = FPUtils[type].mantissa_width()
    var mask1 = _frexp_mask1[simd_width, type]()
    var mask2 = _frexp_mask2[simd_width, type]()
    var x_int = x.to_bits()
    var selector = x != zero
    var exp = selector.select(
        (((mask1 & x_int) >> mantissa_width) - exponent_bias).cast[type](),
        zero,
    )
    var frac = selector.select(T(from_bits=x_int & ~mask1 | mask2), zero)
    return StaticTuple[size=2](frac, exp)


# ===----------------------------------------------------------------------=== #
# log
# ===----------------------------------------------------------------------=== #


@always_inline
fn _log_base[
    type: DType, simd_width: Int, //, base: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Performs elementwise log of a SIMD vector with a specific base.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.
        base: The logarithm base.

    Args:
        x: Vector to perform logarithm operation on.

    Returns:
        Vector containing result of performing logarithm on x.
    """
    # Based on the Cephes approximation.
    alias sqrt2_div_2 = 0.70710678118654752440

    constrained[base == 2 or base == 27, "input base must be either 2 or 27"]()

    var frexp_result = frexp(x)
    var x1 = frexp_result[0]
    var exp = frexp_result[1]
    exp = (x1 < sqrt2_div_2).select(exp - 1, exp)
    x1 = (x1 < sqrt2_div_2).select(x1 + x1, x1) - 1

    var x2 = x1 * x1
    var x3 = x2 * x1

    var y = (
        polynomial_evaluate[
            List[SIMD[type, simd_width]](
                3.3333331174e-1,
                -2.4999993993e-1,
                2.0000714765e-1,
                -1.6668057665e-1,
                1.4249322787e-1,
                -1.2420140846e-1,
                1.1676998740e-1,
                -1.1514610310e-1,
                7.0376836292e-2,
            ),
        ](x1)
        * x3
    )
    y = x1 + x2.fma(-0.5, y)

    # TODO: fix this hack
    @parameter
    if base == 27:  # Natural log
        alias ln2 = 0.69314718055994530942
        y = exp.fma(ln2, y)
    else:
        alias log2e = 1.4426950408889634073599
        y = y.fma(log2e, exp)
    return (x == 0).select(Scalar[type].MIN, (x > 0).select(y, nan[type]()))


@always_inline
fn log(x: SIMD) -> __type_of(x):
    """Performs elementwise natural log (base E) of a SIMD vector.

    Args:
        x: Vector to perform logarithm operation on.

    Returns:
        Vector containing result of performing natural log base E on x.
    """

    @parameter
    if is_nvidia_gpu():
        alias ln2 = 0.69314718055966295651160180568695068359375

        @parameter
        if sizeof[x.type]() < sizeof[DType.float32]():
            return log(x.cast[DType.float32]()).cast[x.type]()
        elif x.type is DType.float32:
            return (
                _call_ptx_intrinsic[
                    instruction="lg2.approx.f32", constraints="=f,f"
                ](x)
                * ln2
            )

    return _log_base[27](x)


# ===----------------------------------------------------------------------=== #
# log2
# ===----------------------------------------------------------------------=== #


@always_inline
fn log2(x: SIMD) -> __type_of(x):
    """Performs elementwise log (base 2) of a SIMD vector.

    Args:
        x: Vector to perform logarithm operation on.

    Returns:
        Vector containing result of performing log base 2 on x.
    """

    @parameter
    if is_nvidia_gpu():

        @parameter
        if sizeof[x.type]() < sizeof[DType.float32]():
            return log2(x.cast[DType.float32]()).cast[x.type]()
        elif x.type is DType.float32:
            return _call_ptx_intrinsic[
                instruction="lg2.approx.f32", constraints="=f,f"
            ](x)

    return _log_base[2](x)


# ===----------------------------------------------------------------------=== #
# copysign
# ===----------------------------------------------------------------------=== #


@always_inline
fn copysign[
    type: DType, simd_width: Int, //
](magnitude: SIMD[type, simd_width], sign: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Returns a value with the magnitude of the first operand and the sign of
    the second operand.

    Constraints:
        The type of the input must be numeric.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        magnitude: The magnitude to use.
        sign: The sign to copy.

    Returns:
        Copies the sign from sign to magnitude.
    """

    @parameter
    if type.is_integral():
        var mag_abs = abs(magnitude)
        return (sign > 0).select(mag_abs, -mag_abs)
    else:
        constrained[type.is_numeric(), "operands must be a numeric type"]()
        return llvm_intrinsic[
            "llvm.copysign", SIMD[type, simd_width], has_side_effect=False
        ](magnitude, sign)


# ===----------------------------------------------------------------------=== #
# erf
# ===----------------------------------------------------------------------=== #


@always_inline
fn erf[
    type: DType, simd_width: Int, //
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Performs the elementwise Erf on a SIMD vector.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
                Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector to perform elementwise Erf on.

    Returns:
        The result of the elementwise Erf operation.
    """
    constrained[type.is_floating_point(), "must be a floating point value"]()
    var x_abs = abs(x)

    var r_large = polynomial_evaluate[
        List[SIMD[type, simd_width]](
            1.28717512e-1,
            6.34846687e-1,
            1.06777847e-1,
            -2.42545605e-2,
            3.88393435e-3,
            -3.83208680e-4,
            1.72948930e-5,
        ),
    ](min(x_abs, 3.925))

    r_large = r_large.fma(x_abs, x_abs)
    r_large = copysign(1 - exp(-r_large), x)

    var r_small = polynomial_evaluate[
        List[SIMD[type, simd_width]](
            1.28379151e-1,
            -3.76124859e-1,
            1.12818025e-1,
            -2.67667342e-2,
            4.99339588e-3,
            -5.99104969e-4,
        ),
    ](x_abs * x_abs).fma(x, x)

    return (x_abs > 0.921875).select[type](r_large, r_small)


# ===----------------------------------------------------------------------=== #
# tanh
# ===----------------------------------------------------------------------=== #


@always_inline
fn tanh[
    type: DType, simd_width: Int, //
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Performs elementwise evaluation of the tanh function.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: The vector to perform the elementwise tanh on.

    Returns:
        The result of the elementwise tanh operation.
    """

    constrained[
        type.is_floating_point(), "the input type must be floating point"
    ]()

    @parameter
    if is_nvidia_gpu():
        alias instruction = "tanh.approx.f32"

        @parameter
        if sizeof[type]() < sizeof[DType.float32]():
            return _call_ptx_intrinsic[
                instruction=instruction, constraints="=f,f"
            ](x.cast[DType.float32]()).cast[type]()
        elif type is DType.float32:
            return _call_ptx_intrinsic[
                instruction=instruction, constraints="=f,f"
            ](x)

    var xc = x.clamp(-9, 9)
    var x_squared = xc * xc

    var numerator = xc * polynomial_evaluate[
        List[SIMD[type, simd_width]](
            4.89352455891786e-03,
            6.37261928875436e-04,
            1.48572235717979e-05,
            5.12229709037114e-08,
            -8.60467152213735e-11,
            2.00018790482477e-13,
            -2.76076847742355e-16,
        ),
    ](x_squared)

    var denominator = polynomial_evaluate[
        List[SIMD[type, simd_width]](
            4.89352518554385e-03,
            2.26843463243900e-03,
            1.18534705686654e-04,
            1.19825839466702e-06,
        ),
    ](x_squared)

    return numerator / denominator


# ===----------------------------------------------------------------------=== #
# isclose
# ===----------------------------------------------------------------------=== #


# TODO: control symmetric behavior with flag so we can be compatible with Python
@always_inline
fn isclose[
    type: DType, simd_width: Int
](
    a: SIMD[type, simd_width],
    b: SIMD[type, simd_width],
    *,
    atol: Scalar[type] = 1e-08,
    rtol: Scalar[type] = 1e-05,
    equal_nan: Bool = False,
) -> SIMD[DType.bool, simd_width]:
    """Checks if the two input values are numerically within a tolerance.

    When the type is integral, then equality is checked. When the type is
    floating point, then this checks if the two input values are numerically the
    close using the $abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol)$
    formula.

    Unlike Pythons's `math.isclose`, this implementation is symmetric. I.e.
    `isclose(a,b) == isclose(b,a)`.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        a: The first value to compare.
        b: The second value to compare.
        atol: The absolute tolerance.
        rtol: The relative tolerance.
        equal_nan: Whether to treat nans as equal.

    Returns:
        A boolean vector where a and b are equal within the specified tolerance.
    """

    constrained[
        a.type is DType.bool or a.type.is_numeric(),
        "input type must be boolean, integral, or floating-point",
    ]()

    @parameter
    if a.type is DType.bool or a.type.is_integral():
        return a == b
    else:
        var both_nan = isnan(a) & isnan(b)
        if equal_nan and all(both_nan):
            return True

        var res = (a == b) | (
            isfinite(a)
            & isfinite(b)
            & (
                abs(a - b)
                <= max(__type_of(a)(atol), rtol * max(abs(a), abs(b)))
            )
        )

        return res | both_nan if equal_nan else res


# ===----------------------------------------------------------------------=== #
# iota
# ===----------------------------------------------------------------------=== #


# TODO: Remove this when `iota` works at compile-time
fn _compile_time_iota[type: DType, simd_width: Int]() -> SIMD[type, simd_width]:
    constrained[
        type.is_integral(),
        "_compile_time_iota can only be used with integer types.",
    ]()
    var a = SIMD[type, simd_width](0)
    for i in range(simd_width):
        a[i] = i
    return a


@always_inline
fn iota[
    type: DType, simd_width: Int
](offset: Scalar[type] = 0) -> SIMD[type, simd_width]:
    """Creates a SIMD vector containing an increasing sequence, starting from
    offset.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        offset: The value to start the sequence at. Default is zero.

    Returns:
        An increasing sequence of values, starting from offset.
    """

    @parameter
    if simd_width == 1:
        return offset
    elif type.is_integral():
        var step = llvm_intrinsic[
            "llvm.stepvector",
            SIMD[type, simd_width],
            has_side_effect=False,
        ]()
        return step + offset
    else:
        var it = llvm_intrinsic[
            "llvm.stepvector",
            SIMD[DType.index, simd_width],
            has_side_effect=False,
        ]()
        return it.cast[type]() + offset


fn iota[
    type: DType
](buff: UnsafePointer[Scalar[type]], len: Int, offset: Int = 0):
    """Fill the buffer with numbers ranging from offset to offset + len - 1,
    spaced by 1.

    The function doesn't return anything, the buffer is updated inplace.

    Parameters:
        type: DType of the underlying data.

    Args:
        buff: The buffer to fill.
        len: The length of the buffer to fill.
        offset: The value to fill at index 0.
    """
    alias simd_width = simdwidthof[type]()
    var vector_end = align_down(len, simd_width)
    for i in range(0, vector_end, simd_width):
        buff.store(i, iota[type, simd_width](i + offset))
    for i in range(vector_end, len):
        buff.store(i, i + offset)


fn iota[type: DType, //](mut v: List[Scalar[type], *_], offset: Int = 0):
    """Fill a list with consecutive numbers starting from the specified offset.

    Parameters:
        type: DType of the underlying data.

    Args:
        v: The list to fill with numbers.
        offset: The starting value to fill at index 0.
    """
    iota(v.data, len(v), offset)


fn iota(mut v: List[Int, *_], offset: Int = 0):
    """Fill a list with consecutive numbers starting from the specified offset.

    Args:
        v: The list to fill with numbers.
        offset: The starting value to fill at index 0.
    """
    var buff = v.data.bitcast[Scalar[DType.index]]()
    iota(buff, len(v), offset=offset)


# ===----------------------------------------------------------------------=== #
# fma
# ===----------------------------------------------------------------------=== #


@always_inline
fn fma(a: Int, b: Int, c: Int) -> Int:
    """Performs `fma` (fused multiply-add) on the inputs.

    The result is `(a * b) + c`.

    Args:
        a: The first input.
        b: The second input.
        c: The third input.

    Returns:
        `(a * b) + c`.
    """
    return a * b + c


@always_inline
fn fma(a: UInt, b: UInt, c: UInt) -> UInt:
    """Performs `fma` (fused multiply-add) on the inputs.

    The result is `(a * b) + c`.

    Args:
        a: The first input.
        b: The second input.
        c: The third input.

    Returns:
        `(a * b) + c`.
    """
    return a * b + c


@always_inline("nodebug")
fn fma[
    type: DType, simd_width: Int
](
    a: SIMD[type, simd_width],
    b: SIMD[type, simd_width],
    c: SIMD[type, simd_width],
) -> SIMD[type, simd_width]:
    """Performs elementwise `fma` (fused multiply-add) on the inputs.

    Each element in the result SIMD vector is $(A_i * B_i) + C_i$, where $A_i$,
    $B_i$ and $C_i$ are elements at index $i$ in a, b, and c respectively.

    Parameters:
        type: The `dtype` of the input SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        a: The first vector of inputs.
        b: The second vector of inputs.
        c: The third vector of inputs.

    Returns:
        Elementwise `fma` of a, b and c.
    """
    return a.fma(b, c)


# ===----------------------------------------------------------------------=== #
# align_down
# ===----------------------------------------------------------------------=== #


@always_inline
fn align_down(value: Int, alignment: Int) -> Int:
    """Returns the closest multiple of alignment that is less than or equal to
    value.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        Closest multiple of the alignment that is less than or equal to the
        input value. In other words, floor(value / alignment) * alignment.
    """
    debug_assert(alignment != 0, "zero alignment")
    return (value // alignment) * alignment


@always_inline
fn align_down(value: UInt, alignment: UInt) -> UInt:
    """Returns the closest multiple of alignment that is less than or equal to
    value.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        Closest multiple of the alignment that is less than or equal to the
        input value. In other words, floor(value / alignment) * alignment.
    """
    debug_assert(alignment != 0, "zero alignment")
    return (value // alignment) * alignment


# ===----------------------------------------------------------------------=== #
# align_up
# ===----------------------------------------------------------------------=== #


@always_inline
fn align_up(value: Int, alignment: Int) -> Int:
    """Returns the closest multiple of alignment that is greater than or equal
    to value.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        Closest multiple of the alignment that is greater than or equal to the
        input value. In other words, ceiling(value / alignment) * alignment.
    """
    debug_assert(alignment != 0, "zero alignment")
    return ceildiv(value, alignment) * alignment


@always_inline
fn align_up(value: UInt, alignment: UInt) -> UInt:
    """Returns the closest multiple of alignment that is greater than or equal
    to value.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        Closest multiple of the alignment that is greater than or equal to the
        input value. In other words, ceiling(value / alignment) * alignment.
    """
    debug_assert(alignment != 0, "zero alignment")
    return ceildiv(value, alignment) * alignment


# ===----------------------------------------------------------------------=== #
# acos
# ===----------------------------------------------------------------------=== #


fn acos(x: SIMD) -> __type_of(x):
    """Computes the `acos` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `acos` of the input.
    """
    return _call_libm["acos"](x)


# ===----------------------------------------------------------------------=== #
# asin
# ===----------------------------------------------------------------------=== #


fn asin(x: SIMD) -> __type_of(x):
    """Computes the `asin` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `asin` of the input.
    """
    return _call_libm["asin"](x)


# ===----------------------------------------------------------------------=== #
# atan
# ===----------------------------------------------------------------------=== #


fn atan(x: SIMD) -> __type_of(x):
    """Computes the `atan` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `atan` of the input.
    """
    return _call_libm["atan"](x)


# ===----------------------------------------------------------------------=== #
# atan2
# ===----------------------------------------------------------------------=== #


fn atan2[
    type: DType, simd_width: Int, //
](y: SIMD[type, simd_width], x: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Computes the `atan2` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        y: The first input argument.
        x: The second input argument.

    Returns:
        The `atan2` of the inputs.
    """

    @always_inline("nodebug")
    @parameter
    fn _float32_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: SIMD[lhs_type, 1], arg1: SIMD[rhs_type, 1]) -> SIMD[result_type, 1]:
        return _external_call_const["atan2f", SIMD[result_type, 1]](arg0, arg1)

    @always_inline("nodebug")
    @parameter
    fn _float64_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: SIMD[lhs_type, 1], arg1: SIMD[rhs_type, 1]) -> SIMD[result_type, 1]:
        return _external_call_const["atan2", SIMD[result_type, 1]](arg0, arg1)

    constrained[type.is_floating_point(), "input type must be floating point"]()

    @parameter
    if type is DType.float64:
        return _simd_apply[_float64_dispatch, type, simd_width](y, x)
    else:
        return _simd_apply[_float32_dispatch, type, simd_width](y, x)


# ===----------------------------------------------------------------------=== #
# cos
# ===----------------------------------------------------------------------=== #


fn cos[
    type: DType, simd_width: Int, //
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Computes the `cos` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `cos` of the input.
    """

    @parameter
    if is_nvidia_gpu() and sizeof[type]() <= sizeof[DType.float32]():
        return _call_ptx_intrinsic[
            instruction="cos.approx.ftz.f32", constraints="=f,f"
        ](x)
    elif is_amd_gpu():
        return llvm_intrinsic["llvm.cos", __type_of(x), has_side_effect=False](
            x
        )
    else:
        return _call_libm["cos"](x)


# ===----------------------------------------------------------------------=== #
# sin
# ===----------------------------------------------------------------------=== #


fn sin[
    type: DType, simd_width: Int, //
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Computes the `sin` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `sin` of the input.
    """

    @parameter
    if is_nvidia_gpu() and sizeof[type]() <= sizeof[DType.float32]():
        return _call_ptx_intrinsic[
            instruction="sin.approx.ftz.f32", constraints="=f,f"
        ](x)
    elif is_amd_gpu():
        return llvm_intrinsic["llvm.sin", __type_of(x), has_side_effect=False](
            x
        )
    else:
        return _call_libm["sin"](x)


# ===----------------------------------------------------------------------=== #
# tan
# ===----------------------------------------------------------------------=== #


fn tan(x: SIMD) -> __type_of(x):
    """Computes the `tan` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `tan` of the input.
    """
    return _call_libm["tan"](x)


# ===----------------------------------------------------------------------=== #
# acosh
# ===----------------------------------------------------------------------=== #


fn acosh(x: SIMD) -> __type_of(x):
    """Computes the `acosh` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `acosh` of the input.
    """
    return _call_libm["acosh"](x)


# ===----------------------------------------------------------------------=== #
# asinh
# ===----------------------------------------------------------------------=== #


fn asinh(x: SIMD) -> __type_of(x):
    """Computes the `asinh` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `asinh` of the input.
    """
    return _call_libm["asinh"](x)


# ===----------------------------------------------------------------------=== #
# atanh
# ===----------------------------------------------------------------------=== #


fn atanh(x: SIMD) -> __type_of(x):
    """Computes the `atanh` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `atanh` of the input.
    """
    return _call_libm["atanh"](x)


# ===----------------------------------------------------------------------=== #
# cosh
# ===----------------------------------------------------------------------=== #


fn cosh(x: SIMD) -> __type_of(x):
    """Computes the `cosh` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `cosh` of the input.
    """
    return _call_libm["cosh"](x)


# ===----------------------------------------------------------------------=== #
# sinh
# ===----------------------------------------------------------------------=== #


fn sinh(x: SIMD) -> __type_of(x):
    """Computes the `sinh` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `sinh` of the input.
    """
    return _call_libm["sinh"](x)


# ===----------------------------------------------------------------------=== #
# expm1
# ===----------------------------------------------------------------------=== #


@always_inline
fn expm1(x: SIMD) -> __type_of(x):
    """Computes the `expm1` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `expm1` of the input.
    """
    return _call_libm["expm1"](x)


# ===----------------------------------------------------------------------=== #
# log10
# ===----------------------------------------------------------------------=== #


fn log10(x: SIMD) -> __type_of(x):
    """Computes the `log10` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `log10` of the input.
    """

    @parameter
    if is_nvidia_gpu():
        alias log10_2 = 0.301029995663981195213738894724493027

        @parameter
        if sizeof[x.type]() < sizeof[DType.float32]():
            return log10(x.cast[DType.float32]()).cast[x.type]()
        elif x.type is DType.float32:
            return (
                _call_ptx_intrinsic[
                    instruction="lg2.approx.f32", constraints="=f,f"
                ](x)
                * log10_2
            )
    elif is_amd_gpu():
        return llvm_intrinsic[
            "llvm.log10", __type_of(x), has_side_effect=False
        ](x)

    return _call_libm["log10"](x)


# ===----------------------------------------------------------------------=== #
# log1p
# ===----------------------------------------------------------------------=== #


fn log1p(x: SIMD) -> __type_of(x):
    """Computes the `log1p` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `log1p` of the input.
    """
    return _call_libm["log1p"](x)


# ===----------------------------------------------------------------------=== #
# logb
# ===----------------------------------------------------------------------=== #


fn logb(x: SIMD) -> __type_of(x):
    """Computes the `logb` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `logb` of the input.
    """
    return _call_libm["logb"](x)


# ===----------------------------------------------------------------------=== #
# cbrt
# ===----------------------------------------------------------------------=== #


fn cbrt(x: SIMD) -> __type_of(x):
    """Computes the `cbrt` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `cbrt` of the input.
    """
    return _call_libm["cbrt"](x)


# ===----------------------------------------------------------------------=== #
# hypot
# ===----------------------------------------------------------------------=== #


# TODO: implement for variadic inputs as Python.
fn hypot[
    type: DType, simd_width: Int, //
](arg0: SIMD[type, simd_width], arg1: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Computes the `hypot` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg0: The first input argument.
        arg1: The second input argument.

    Returns:
        The `hypot` of the inputs.
    """

    @always_inline("nodebug")
    @parameter
    fn _float32_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: SIMD[lhs_type, 1], arg1: SIMD[rhs_type, 1]) -> SIMD[result_type, 1]:
        return _external_call_const["hypotf", SIMD[result_type, 1]](arg0, arg1)

    @always_inline("nodebug")
    @parameter
    fn _float64_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: SIMD[lhs_type, 1], arg1: SIMD[rhs_type, 1]) -> SIMD[result_type, 1]:
        return _external_call_const["hypot", SIMD[result_type, 1]](arg0, arg1)

    constrained[type.is_floating_point(), "input type must be floating point"]()

    @parameter
    if type is DType.float64:
        return _simd_apply[_float64_dispatch, type, simd_width](arg0, arg1)
    return _simd_apply[_float32_dispatch, type, simd_width](arg0, arg1)


# ===----------------------------------------------------------------------=== #
# erfc
# ===----------------------------------------------------------------------=== #


fn erfc(x: SIMD) -> __type_of(x):
    """Computes the `erfc` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `erfc` of the input.
    """
    return _call_libm["erfc"](x)


# ===----------------------------------------------------------------------=== #
# lgamma
# ===----------------------------------------------------------------------=== #


fn lgamma(x: SIMD) -> __type_of(x):
    """Computes the `lgamma` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The `lgamma` of the input.
    """
    return _call_libm["lgamma"](x)


# ===----------------------------------------------------------------------=== #
# gamma
# ===----------------------------------------------------------------------=== #


fn gamma(x: SIMD) -> __type_of(x):
    """Computes the Gamma of the input.

    For details, see https://en.wikipedia.org/wiki/Gamma_function.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input argument.

    Returns:
        The Gamma function evaluated at the input.
    """
    return _call_libm["tgamma"](x)


# ===----------------------------------------------------------------------=== #
# remainder
# ===----------------------------------------------------------------------=== #


fn remainder[
    type: DType, simd_width: Int, //
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Computes the `remainder` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: The first input argument.
        y: The second input argument.

    Returns:
        The `remainder` of the inputs.
    """

    @always_inline("nodebug")
    @parameter
    fn _float32_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: SIMD[lhs_type, 1], arg1: SIMD[rhs_type, 1]) -> SIMD[result_type, 1]:
        return _external_call_const["remainderf", SIMD[result_type, 1]](
            arg0, arg1
        )

    @always_inline("nodebug")
    @parameter
    fn _float64_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: SIMD[lhs_type, 1], arg1: SIMD[rhs_type, 1]) -> SIMD[result_type, 1]:
        return _external_call_const["remainder", SIMD[result_type, 1]](
            arg0, arg1
        )

    constrained[type.is_floating_point(), "input type must be floating point"]()

    @parameter
    if type is DType.float64:
        return _simd_apply[_float64_dispatch, type, simd_width](x, y)
    return _simd_apply[_float32_dispatch, type, simd_width](x, y)


# ===----------------------------------------------------------------------=== #
# j0
# ===----------------------------------------------------------------------=== #


fn j0(x: SIMD) -> __type_of(x):
    """Computes the Bessel function of the first kind of order 0 for each input
    value.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input vector.

    Returns:
        A vector containing the computed value for each value in the input.
    """
    return _call_libm["j0"](x)


# ===----------------------------------------------------------------------=== #
# j1
# ===----------------------------------------------------------------------=== #


fn j1(x: SIMD) -> __type_of(x):
    """Computes the Bessel function of the first kind of order 1 for each input
    value.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input vector.

    Returns:
        A vector containing the computed value for each value in the input.
    """
    return _call_libm["j1"](x)


# ===----------------------------------------------------------------------=== #
# y0
# ===----------------------------------------------------------------------=== #


fn y0(x: SIMD) -> __type_of(x):
    """Computes the Bessel function of the second kind of order 0 for each input
    value.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input vector.

    Returns:
        A vector containing the computed value for each value in the input.
    """
    return _call_libm["y0"](x)


# ===----------------------------------------------------------------------=== #
# y1
# ===----------------------------------------------------------------------=== #


fn y1(x: SIMD) -> __type_of(x):
    """Computes the Bessel function of the second kind of order 1 for each input
    value.

    Constraints:
        The input must be a floating-point type.

    Args:
        x: The input vector.

    Returns:
        A vector containing the computed value for each value in the input.
    """
    return _call_libm["y1"](x)


# ===----------------------------------------------------------------------=== #
# scalb
# ===----------------------------------------------------------------------=== #


fn scalb[
    type: DType, simd_width: Int, //
](arg0: SIMD[type, simd_width], arg1: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Computes the `scalb` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg0: The first input argument.
        arg1: The second input argument.

    Returns:
        The `scalb` of the inputs.
    """

    @always_inline("nodebug")
    @parameter
    fn _float32_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: SIMD[lhs_type, 1], arg1: SIMD[rhs_type, 1]) -> SIMD[result_type, 1]:
        return _external_call_const["scalbf", SIMD[result_type, 1]](arg0, arg1)

    @always_inline("nodebug")
    @parameter
    fn _float64_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: SIMD[lhs_type, 1], arg1: SIMD[rhs_type, 1]) -> SIMD[result_type, 1]:
        return _external_call_const["scalb", SIMD[result_type, 1]](arg0, arg1)

    constrained[type.is_floating_point(), "input type must be floating point"]()

    @parameter
    if type is DType.float64:
        return _simd_apply[_float64_dispatch, type, simd_width](arg0, arg1)
    return _simd_apply[_float32_dispatch, type, simd_width](arg0, arg1)


# ===----------------------------------------------------------------------=== #
# gcd
# ===----------------------------------------------------------------------=== #


fn gcd(m: Int, n: Int, /) -> Int:
    """Compute the greatest common divisor of two integers.

    Args:
        m: The first integer.
        n: The second integrer.

    Returns:
        The greatest common divisor of the two integers.
    """
    var u = abs(m)
    var v = abs(n)
    if u == 0:
        return v
    if v == 0:
        return u

    var uz = count_trailing_zeros(u)
    var vz = count_trailing_zeros(v)
    var shift = min(uz, vz)
    u >>= shift
    while True:
        v >>= vz
        var diff = v - u
        if diff == 0:
            break
        u, v = min(u, v), abs(diff)
        vz = count_trailing_zeros(diff)
    return u << shift


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
fn gcd(l: List[Int, *_], /) -> Int:
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


fn lcm(m: Int, n: Int, /) -> Int:
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
fn lcm(l: List[Int, *_], /) -> Int:
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


# ===----------------------------------------------------------------------=== #
# modf
# ===----------------------------------------------------------------------=== #


fn modf(x: SIMD) -> Tuple[__type_of(x), __type_of(x)]:
    """Computes the integral and fractional part of the value.

    Args:
        x: The input value.

    Returns:
        A tuple containing the integral and fractional part of the value.
    """
    return _modf(x)


# ===----------------------------------------------------------------------=== #
# ulp
# ===----------------------------------------------------------------------=== #


@always_inline
fn ulp[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Computes the ULP (units of last place) or (units of least precision) of
    the number.

    Constraints:
        The element type of the inpiut must be a floating-point type.

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
        simd_width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector input.

    Returns:
        The ULP of x.
    """

    constrained[type.is_floating_point(), "the type must be floating point"]()

    var nan_mask = isnan(x)
    var xabs = abs(x)
    var inf_mask = isinf(xabs)
    alias inf_val = SIMD[type, simd_width](inf[type]())
    var x2 = nextafter(xabs, inf_val)
    var x2_inf_mask = isinf(x2)

    return nan_mask.select(
        x,
        inf_mask.select(
            xabs,
            x2_inf_mask.select(xabs - nextafter(xabs, -inf_val), x2 - xabs),
        ),
    )


# ===----------------------------------------------------------------------=== #
# factorial
# ===----------------------------------------------------------------------=== #


# TODO: implement for IntLiteral
@always_inline
fn factorial(n: Int) -> Int:
    """Computes the factorial of the integer.

    Args:
        n: The input value. Must be non-negative.

    Returns:
        The factorial of the input. Results are undefined for negative inputs.
    """
    alias table = StaticTuple[Int, 21](
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    )
    debug_assert(0 <= n <= 20, "input value causes an overflow")
    return table[n]


# ===----------------------------------------------------------------------=== #
# clamp
# ===----------------------------------------------------------------------=== #


fn clamp(
    val: Int, lower_bound: __type_of(val), upper_bound: __type_of(val)
) -> __type_of(val):
    """Clamps the integer value vector to be in a certain range.

    Args:
        val: The value to clamp.
        lower_bound: Minimum of the range to clamp to.
        upper_bound: Maximum of the range to clamp to.

    Returns:
        An integer clamped to be within lower_bound and upper_bound.
    """
    return max(min(val, upper_bound), lower_bound)


fn clamp(
    val: UInt, lower_bound: __type_of(val), upper_bound: __type_of(val)
) -> __type_of(val):
    """Clamps the integer value vector to be in a certain range.

    Args:
        val: The value to clamp.
        lower_bound: Minimum of the range to clamp to.
        upper_bound: Maximum of the range to clamp to.

    Returns:
        An integer clamped to be within lower_bound and upper_bound.
    """
    return max(min(val, upper_bound), lower_bound)


fn clamp(
    val: SIMD, lower_bound: __type_of(val), upper_bound: __type_of(val)
) -> __type_of(val):
    """Clamps the values in a SIMD vector to be in a certain range.

    Clamp cuts values in the input SIMD vector off at the upper bound and
    lower bound values. For example,  SIMD vector `[0, 1, 2, 3]` clamped to
    a lower bound of 1 and an upper bound of 2 would return `[1, 1, 2, 2]`.

    Args:
        val: The value to clamp.
        lower_bound: Minimum of the range to clamp to.
        upper_bound: Maximum of the range to clamp to.

    Returns:
        A SIMD vector containing x clamped to be within lower_bound and
        upper_bound.
    """
    return val.clamp(lower_bound, upper_bound)


# ===----------------------------------------------------------------------=== #
# utilities
# ===----------------------------------------------------------------------=== #


fn _type_is_libm_supported(type: DType) -> Bool:
    return type in (DType.float32, DType.float64) or type.is_integral()


fn _call_libm[
    func_name: StringLiteral,
    arg_type: DType,
    simd_width: Int,
    *,
    result_type: DType = arg_type,
](arg: SIMD[arg_type, simd_width]) -> SIMD[result_type, simd_width]:
    constrained[
        arg_type.is_floating_point(), "argument type must be floating point"
    ]()
    constrained[
        arg_type == result_type, "the argument type must match the result type"
    ]()
    constrained[
        not is_nvidia_gpu(),
        "the libm operation is not available on the CUDA target",
    ]()

    @parameter
    if not _type_is_libm_supported(arg_type):
        # Coerse to f32 if the value is not representable by libm.
        return _call_libm_impl[func_name, result_type = DType.float32](
            arg.cast[DType.float32]()
        ).cast[result_type]()

    return _call_libm_impl[func_name, result_type=result_type](arg)


fn _call_libm_impl[
    func_name: StringLiteral,
    arg_type: DType,
    simd_width: Int,
    *,
    result_type: DType = arg_type,
](arg: SIMD[arg_type, simd_width]) -> SIMD[result_type, simd_width]:
    alias libm_name = func_name if arg_type is DType.float64 else func_name + "f"

    var res = SIMD[result_type, simd_width]()

    @parameter
    for i in range(simd_width):
        res[i] = _external_call_const[libm_name, Scalar[result_type]](arg[i])

    return res


fn _call_ptx_intrinsic_scalar[
    type: DType, //,
    *,
    instruction: StringLiteral,
    constraints: StringLiteral,
](arg: Scalar[type]) -> Scalar[type]:
    return inlined_assembly[
        instruction + " $0, $1;",
        Scalar[type],
        constraints=constraints,
        has_side_effect=False,
    ](arg)


fn _call_ptx_intrinsic_scalar[
    type: DType, //,
    *,
    instruction: StringLiteral,
    constraints: StringLiteral,
](arg0: Scalar[type], arg1: Scalar[type]) -> Scalar[type]:
    return inlined_assembly[
        instruction + " $0, $1, $2;",
        Scalar[type],
        constraints=constraints,
        has_side_effect=False,
    ](arg0, arg1)


fn _call_ptx_intrinsic[
    type: DType,
    simd_width: Int, //,
    *,
    instruction: StringLiteral,
    constraints: StringLiteral,
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    @parameter
    if simd_width == 1:
        return _call_ptx_intrinsic_scalar[
            instruction=instruction, constraints=constraints
        ](arg[0])

    var res = SIMD[type, simd_width]()

    @parameter
    for i in range(simd_width):
        res[i] = _call_ptx_intrinsic_scalar[
            instruction=instruction, constraints=constraints
        ](arg[i])
    return res


fn _call_ptx_intrinsic[
    type: DType,
    simd_width: Int, //,
    *,
    scalar_instruction: StringLiteral,
    vector2_instruction: StringLiteral,
    scalar_constraints: StringLiteral,
    vector_constraints: StringLiteral,
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    @parameter
    if simd_width == 1:
        return _call_ptx_intrinsic_scalar[
            instruction=scalar_instruction, constraints=scalar_constraints
        ](arg[0])

    var res = SIMD[type, simd_width]()

    @parameter
    for i in range(0, simd_width, 2):
        res = res.insert[offset=i](
            inlined_assembly[
                vector2_instruction + " $0, $1;",
                SIMD[type, 2],
                constraints=vector_constraints,
                has_side_effect=False,
            ](arg.slice[2, offset=i]())
        )

    return res


fn _call_ptx_intrinsic[
    type: DType,
    simd_width: Int, //,
    *,
    scalar_instruction: StringLiteral,
    vector2_instruction: StringLiteral,
    scalar_constraints: StringLiteral,
    vector_constraints: StringLiteral,
](arg0: SIMD[type, simd_width], arg1: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    @parameter
    if simd_width == 1:
        return _call_ptx_intrinsic_scalar[
            instruction=scalar_instruction, constraints=scalar_constraints
        ](arg0[0], arg1[0])

    var res = SIMD[type, simd_width]()

    @parameter
    for i in range(0, simd_width, 2):
        res = res.insert[offset=i](
            inlined_assembly[
                vector2_instruction + " $0, $1; $2;",
                SIMD[type, 2],
                constraints=vector_constraints,
                has_side_effect=False,
            ](arg0.slice[2, offset=i](), arg1.slice[2, offset=i]())
        )

    return res


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
            return Self(ceil(self.re), ceil(self.im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __ceil__(self) -> Self:
        """Return the ceiling of the Int value, which is itself.

        Returns:
            The Int value itself.
        """
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
            return Self(floor(self.re), floor(self.im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __floor__(self) -> Self:
        """Return the floor of the Int value, which is itself.

        Returns:
            The Int value itself.
        """
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

        fn __ceildiv__(self, denominator: Self) -> Self:
            return -(self.x // -denominator.x)
    ```
    """

    fn __ceildiv__(self, denominator: Self) -> Self:
        """Return the rounded-up result of dividing self by denominator.

        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.
        """
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

        fn __ceildiv__(self, denominator: Self) raises -> Self:
            return -(self.x // -denominator.x)
    ```
    """

    fn __ceildiv__(self, denominator: Self) raises -> Self:
        """Return the rounded-up result of dividing self by denominator.

        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.
        """
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
            return Self(trunc(self.re), trunc(self.im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __trunc__(self) -> Self:
        """Return the truncated Int value, which is itself.

        Returns:
            The Int value itself.
        """
        ...
