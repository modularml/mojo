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

"""The module contains implementations of activation functions."""

import std.math
from std.utils.numerics import get_accum_type


# ===----------------------------------------------------------------------=== #
# sign
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
def _is_neg[
    dtype: DType, simd_width: SIMDLength
](val: SIMD[dtype, simd_width]) -> SIMD[.bool, simd_width]:
    """Returns True if the input value is negative.

    The value is computed separately for each element in the SIMD vector. For
    unsigned dtypes the result is always a SIMD vector filled with False.

    Parameters:
        dtype: dtype used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        val: The value to check.

    Returns:
        A SIMD value where the element at position `i` is True if the value is
        negative at position `i` and False otherwise.
    """

    comptime if dtype.is_unsigned():
        return SIMD[.bool, simd_width](fill=False)
    return val.lt(0)


@always_inline
def sign[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the sign (0, 1) of the input value.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the sign operation on.

    Returns:
        The result of the sign operation.
    """
    var is_neg_mask = _is_neg(x)
    var is_zero_mask = x.eq(0)
    return is_neg_mask.select[dtype](-1, is_zero_mask.select[dtype](0, 1))


# ===----------------------------------------------------------------------=== #
# elu
# ===----------------------------------------------------------------------=== #


@always_inline
def elu[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the Elu Op using the equation $z if z >= 0 else alpha*(e^z -1)$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x: The value to compute the ELU operation on.

    Returns:
        The result of the ELU operation.
    """
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    return x.ge(0).select(x, std.math.expm1(x))


# ===----------------------------------------------------------------------=== #
# relu
# ===----------------------------------------------------------------------=== #


@always_inline
def relu[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the Relu Op using the equation $max(x, 0)$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the RELU operation on.

    Returns:
        The result of the RELU operation.
    """
    return max(x, 0)


# ===----------------------------------------------------------------------=== #
# relu-n1
# ===----------------------------------------------------------------------=== #


@always_inline
def relu_n1[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the Relu N1 Op using the equation $max(min(x,1),-1)$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the RELU N1 operation on.

    Returns:
        The result of the RELU N1 operation.
    """
    return x.clamp(-1, 1)


# ===----------------------------------------------------------------------=== #
# leaky_relu
# ===----------------------------------------------------------------------=== #


@always_inline
def leaky_relu[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width], negative_slope: Scalar[dtype]) -> SIMD[
    dtype, simd_width
]:
    """Compute the Leaky ReLU using the equation
    $max(x, 0) + negative_slope * min(x, 0)$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x: The value to compute the Leaky ReLU operation on.
        negative_slope: The slope for negative values.

    Constraints:
        Type must be a floating point Dtype.

    Returns:
        The result of the Leaky ReLU operation.
    """
    comptime assert (
        dtype.is_floating_point()
    ), "dtype must be a floating point dtype"
    return x.ge(0).select(x, negative_slope * x)


# ===----------------------------------------------------------------------=== #
# sigmoid
# ===----------------------------------------------------------------------=== #


@always_inline
def sigmoid[
    dtype: DType,
    simd_width: SIMDLength,
    accum: DType = get_accum_type[dtype](),
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the sigmoid activation using the equation $1 / (1 + e^{-x})$.

    The computation is performed in a higher-precision accumulation type (see
    `get_accum_type`) for low-precision inputs and cast back to `dtype`, to
    match the numerics of the graph-level `ops.sigmoid` implementation.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.
        accum: Higher-precision accumulation dtype used internally; defaults
            to `get_accum_type[dtype]()`.

    Args:
        x: The value to compute the sigmoid operation on.

    Constraints:
        Type must be a floating point Dtype.

    Returns:
        The result of the sigmoid operation.
    """
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert accum.is_floating_point(), "accum must be floating point"
    var x_cast = x.cast[accum]()
    return (1 / (1 + std.math.exp(-x_cast))).cast[dtype]()


# ===----------------------------------------------------------------------=== #
# silu
# ===----------------------------------------------------------------------=== #


@always_inline
def silu[
    dtype: DType,
    simd_width: SIMDLength,
    accum: DType = get_accum_type[dtype](),
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the SiLU (Swish) activation using the equation
    $x * sigmoid(x)$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.
        accum: Higher-precision accumulation dtype used internally; defaults
            to `get_accum_type[dtype]()`.

    Args:
        x: The value to compute the SiLU operation on.

    Constraints:
        Type must be a floating point Dtype.

    Returns:
        The result of the SiLU operation.
    """
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert accum.is_floating_point(), "accum must be floating point"
    var x_cast = x.cast[accum]()
    return (x_cast * (1 / (1 + std.math.exp(-x_cast)))).cast[dtype]()


# ===----------------------------------------------------------------------=== #
# gelu
# ===----------------------------------------------------------------------=== #


@always_inline
def gelu[
    dtype: DType,
    simd_width: SIMDLength,
    accum: DType = get_accum_type[dtype](),
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the exact GELU activation using the equation
    $0.5 * x * (1 + erf(x / sqrt(2)))$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.
        accum: Higher-precision accumulation dtype used internally; defaults
            to `get_accum_type[dtype]()`.

    Args:
        x: The value to compute the GELU operation on.

    Constraints:
        Type must be a floating point Dtype.

    Returns:
        The result of the exact GELU operation.
    """
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert accum.is_floating_point(), "accum must be floating point"
    var x_cast = x.cast[accum]()
    return (
        0.5 * x_cast * (1 + std.math.erf(x_cast / 1.4142135623730951))
    ).cast[dtype]()


@always_inline
def gelu_tanh[
    dtype: DType,
    simd_width: SIMDLength,
    accum: DType = get_accum_type[dtype](),
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the tanh approximation of the GELU activation:
    $0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.
        accum: Higher-precision accumulation dtype used internally; defaults
            to `get_accum_type[dtype]()`.

    Args:
        x: The value to compute the tanh-GELU operation on.

    Constraints:
        Type must be a floating point Dtype.

    Returns:
        The result of the tanh-GELU operation.
    """
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert accum.is_floating_point(), "accum must be floating point"
    var x_cast = x.cast[accum]()
    return (
        x_cast
        * 0.5
        * (
            1.0
            + std.math.tanh(
                0.7978845608028654
                * (x_cast + 0.044715 * x_cast * x_cast * x_cast)
            )
        )
    ).cast[dtype]()


@always_inline
def gelu_quick[
    dtype: DType,
    simd_width: SIMDLength,
    accum: DType = get_accum_type[dtype](),
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the quick (sigmoid) approximation of the GELU activation:
    $x * sigmoid(1.702 * x)$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.
        accum: Higher-precision accumulation dtype used internally; defaults
            to `get_accum_type[dtype]()`.

    Args:
        x: The value to compute the quick-GELU operation on.

    Constraints:
        Type must be a floating point Dtype.

    Returns:
        The result of the quick-GELU operation.
    """
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert accum.is_floating_point(), "accum must be floating point"
    var x_cast = x.cast[accum]()
    return (x_cast * (1 / (1 + std.math.exp(-(1.702 * x_cast))))).cast[dtype]()
