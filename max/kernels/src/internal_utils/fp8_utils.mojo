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
"""Shared FP8 quantization utilities.

Provides common functions for FP8 scale computation and quantization
used across fused normalization kernels and standalone quantization kernels.
"""

from std.math import clamp
from std.utils.numerics import isinf, isnan, max_finite, min_finite


@always_inline
def guarded_inv_scale[dt: DType](scale: Scalar[dt]) -> Scalar[dt]:
    """Reciprocal of an FP8 scale factor, guarded to stay finite.

    For a near-zero quant group the dynamic scale `max_abs / fp8_max` underflows
    to a tiny denormal whose reciprocal overflows to +Inf; a later
    `value * inv_scale` is then +Inf (and `0 * Inf = NaN` on a zero lane) before
    the FP8 cast. Gate on `scale` first so the division is skipped for a zero
    scale, and treat a non-finite reciprocal as zero so the (effectively zero)
    group quantizes to a clean FP8 zero.

    Parameters:
        dt: The scale dtype.

    Args:
        scale: The FP8 scale factor (`max_abs / fp8_max`), non-negative.

    Returns:
        `1 / scale`, or `0` when `scale` is zero or its reciprocal is non-finite.
    """
    if scale == 0:
        return 0
    var inv_scale = 1.0 / scale
    if isinf(inv_scale) or isnan(inv_scale):
        return 0
    return inv_scale


@always_inline
def compute_dynamic_fp8_scale[
    out_dtype: DType,
](
    row_max: Scalar,
    scale_ub: Scalar,
) -> Tuple[
    type_of(scale_ub), type_of(row_max)
]:
    """Compute dynamic FP8 scale factor and its reciprocal from a row max.

    Computes scale_factor = min(row_max, scale_ub) / fp8_max and its reciprocal.
    Does not use `math.recip` to avoid a reciprocal approximation that gives up
    too much precision.

    Parameters:
        out_dtype: The FP8 output dtype (float8_e4m3fn or float8_e4m3fnuz).

    Args:
        row_max: Maximum absolute value across the row/group.
        scale_ub: Upper bound to clamp the scale factor.

    Returns:
        A tuple of (scale_factor, scale_factor_recip).
    """
    comptime assert out_dtype.is_float8(), "out_dtype must be float8"

    comptime fp8_max = max_finite[out_dtype]()
    var scale_factor = (
        min(row_max.cast[scale_ub.dtype](), scale_ub)
        / fp8_max.cast[scale_ub.dtype]()
    )
    # Guard the reciprocal against denormal-scale overflow (see
    # `guarded_inv_scale`): a near-zero group's `scale_factor` underflows to a
    # tiny denormal whose `1/scale_factor` overflows to +Inf, NaN-ing the fp8
    # cast on a zero lane (0*Inf). #87813's saturating cast does not help here —
    # it is downstream and turns the +Inf into ±max_finite garbage. Treat a
    # non-finite reciprocal as zero so the group quantizes to a clean fp8 zero.
    var scale_factor_recip = guarded_inv_scale(
        scale_factor.cast[row_max.dtype]()
    )
    return (scale_factor, scale_factor_recip)


@always_inline
def fp8_quantize[
    out_dtype: DType,
    *,
    use_clamp: Bool = True,
](values: SIMD, scale_recip: Scalar[values.dtype]) -> SIMD[
    out_dtype, values.length
]:
    """Quantize values to FP8, clamping to the representable range.

    FP8 quantization must SATURATE out-of-range inputs to [min_finite,
    max_finite]. A scaled value can exceed `max_finite[out_dtype]` (±448 for
    e4m3) whenever the dynamic-scale amax was capped by `scale_ub`
    (`scale_recip = fp8_max / min(amax, scale_ub)`, so values in
    `(scale_ub, amax]` map past fp8_max), or when the input carries an outlier.
    On a raw NVIDIA `cvt` to e4m3 an out-of-range fp32 value produces NaN
    (the cvt is not `satfinite`), which poisons the downstream GEMM and the
    logits. Clamping before the cast makes the cast saturate (the correct FP8
    behavior) and is NaN-safe. AMD already clamped (its cvt was the slow NaN
    path); this default now clamps on NVIDIA too.

    Parameters:
        out_dtype: The FP8 output dtype.
        use_clamp: Whether to clamp to [min_finite, max_finite] before cast.
            Defaults to True on all targets (clamp is required for NaN-safe,
            saturating FP8 quantization). Set False only for a microbench that
            deliberately measures the raw-cast cost.

    Args:
        values: Values to quantize (already normalized as needed, not yet scaled).
        scale_recip: Reciprocal of the FP8 scale factor.

    Returns:
        FP8-quantized values.
    """

    comptime assert out_dtype.is_float8(), "out_dtype must be float8"
    var result = values * scale_recip

    comptime if use_clamp:
        comptime min_val = SIMD[values.dtype, values.length](
            min_finite[out_dtype]()
        )
        comptime max_val = SIMD[values.dtype, values.length](
            max_finite[out_dtype]()
        )
        return clamp(result, min_val, max_val).cast[out_dtype]()
    else:
        return result.cast[out_dtype]()


@always_inline
def cast_saturating[
    in_dtype: DType,
    width: SIMDLength,
    //,
    out_dtype: DType,
](values: SIMD[in_dtype, width]) -> SIMD[out_dtype, width]:
    """Cast to `out_dtype`, saturating to the FP8 representable range first.

    A plain `.cast[float8_e4m3fn]()` of an out-of-range fp32 value produces NaN
    on NVIDIA (the `cvt` is not `satfinite`); FP8 stores must SATURATE instead.
    This helper clamps to [min_finite, max_finite] before the cast when
    `out_dtype` is FP8, and is a plain cast otherwise (no-op clamp for
    bf16/f16). Use it for direct stores to a (possibly-FP8) destination, e.g.
    the MLA RoPE / RMSNorm KV-cache writes where the cache dtype is generic.

    Parameters:
        out_dtype: The destination dtype.

    Args:
        values: Values to cast.

    Returns:
        The values cast to `out_dtype`, saturated if `out_dtype` is FP8.
    """
    comptime if in_dtype == out_dtype or not out_dtype.is_float8():
        return values.cast[out_dtype]()

    comptime min_val = SIMD[values.dtype, values.length](
        min_finite[out_dtype]()
    )
    comptime max_val = SIMD[values.dtype, values.length](
        max_finite[out_dtype]()
    )
    return clamp(values, min_val, max_val).cast[out_dtype]()
