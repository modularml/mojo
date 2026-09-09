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

"""Provides low-level FP4 encode/decode utilities and scale-factor helpers."""

from std.sys._assembly import inlined_assembly
from std.sys import is_nvidia_gpu, bit_width_of, llvm_intrinsic
from std.sys.info import _is_sm_100x_or_newer, _cdna_4_or_newer, align_of
from std.utils.numerics import FPUtils
from std.utils.index import IndexList
from std.memory import bitcast
from layout import Coord, CoordLike, Idx, Layout, LayoutTensor, TileTensor
from std.builtin.simd import _convert_f32_to_float8_ue8m0
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind

comptime SF_ATOM_M = (32, 4)
comptime SF_ATOM_K = 4
comptime SF_MN_GROUP_SIZE: Int = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128
comptime SF_K_GROUP_SIZE[SF_VECTOR_SIZE: Int]: Int = SF_ATOM_K * SF_VECTOR_SIZE

comptime NVFP4_SF_VECTOR_SIZE = 16
comptime MXFP4_SF_VECTOR_SIZE = 32
comptime MXFP8_SF_VECTOR_SIZE = 32

# Reciprocal of E4M3's maxabs. Exact as a literal, so the MXFP8 block-scale
# derivation carries no `V_RCP_F32` rounding and no reliance on LLVM folding
# the intrinsic.
comptime E4M3_MAXABS_RECIP = 1.0 / 448.0

comptime NVFP4_SF_DTYPE = DType.float8_e4m3fn
comptime MXFP4_SF_DTYPE = DType.float8_e8m0fnu
comptime MXFP8_SF_DTYPE = DType.float8_e8m0fnu

comptime FP4_E2M1_MANTISSA_WIDTH = 1
comptime FP4_E2M1_MAX_EXPONENT = 2

# W4A8: E4M3 activations against E2M1 weights, both on E8M0 group-32 scales.
# `kind::mxf8f6f4` encodes the two operand formats independently, so the pair
# reaches the tensor cores without dequantizing the weights first. Global
# memory keeps the weights nibble-packed; the FP4 TMA copy pads them on the way
# in, landing each 16-value group in 16 shared-memory bytes as 8 packed bytes
# followed by an 8-byte gap, which is the form this UMMA kind reads. The values
# themselves stay nibble-packed; it is the padding that makes a K extent span
# one byte per element. The FP4-only kinds are different -- there BOTH operands
# stay packed and unpadded all the way into shared memory.
comptime W4A8_A_DTYPE = DType.float8_e4m3fn
comptime W4A8_B_DTYPE = DType.uint8


def is_w4a8_operand_pair[a_type: DType, b_type: DType]() -> Bool:
    """Reports whether (activations, weights) is the mixed W4A8 pair.

    Directional, for callers that name their operands: activations first,
    weights second. Once inside the kernel the two can arrive in either slot
    (see `block_scaled_operands_compatible`).
    """
    return a_type == W4A8_A_DTYPE and b_type == W4A8_B_DTYPE


def _is_packed_fp4[dtype: DType, other_dtype: DType]() -> Bool:
    """Reports whether this operand is E2M1 that a TMA copy must unpack.

    True only for the mixed W4A8 pair. When both operands are `uint8` the UMMA
    kind is one of the FP4-only ones, which reads nibble-packed shared memory
    directly and must not be unpacked.
    """
    return dtype == W4A8_B_DTYPE and other_dtype == W4A8_A_DTYPE


def block_scaled_operands_compatible[a_type: DType, b_type: DType]() -> Bool:
    """Reports whether A and B may be fed to one block-scaled UMMA.

    Every kind but `kind::mxf8f6f4` requires both operands in the same format;
    that one encodes the two formats independently, so it also admits the W4A8
    pair -- in either slot, since `AB_swapped` hands the weights to A.
    """
    return (
        a_type == b_type
        or is_w4a8_operand_pair[a_type, b_type]()
        or is_w4a8_operand_pair[b_type, a_type]()
    )


comptime E2M1_TO_FLOAT32 = SIMD[.float32, 16](
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@always_inline
def decode_e2m1_to_bf16[
    width: SIMDLength, //
](nibble: SIMD[.uint16, width]) -> SIMD[.bfloat16, width]:
    """Decodes E2M1 nibbles to bfloat16 with branch-free bit arithmetic.

    Maps each 4-bit E2M1 value (`s e1 e0 m0`: bit3 sign, bits2:1 exponent,
    bit0 mantissa) directly to a bfloat16 bit pattern using shifts / masks /
    select, with no data-dependent table gather. This is the vectorizable
    equivalent of indexing `E2M1_TO_FLOAT32` and is **bit-identical** to it:
    all 16 E2M1 values (`{+-0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}`) are
    exactly representable in bfloat16, so the constructed bit pattern equals
    the table entry exactly.

    Construction (bf16 layout `s | 8-bit exp (bias 127) | 7-bit mantissa`):
    let `E` = bits2:1 and `m` = bit0.
    - Normal (`E >= 1`): value `2^(E-1) * (1 + m/2)`, so exponent field
      `E + 126` and mantissa MSB `m` (mantissa field `m << 6`).
    - Subnormal-class (`E == 0`): value `0.5 * m`, i.e. bf16 `0x3F00`
      (`+0.5`) when `m == 1`, else `0x0000` (`+0.0`). Selected branch-free.
    - Sign bit (`s`) is OR'd into bit 15, so `-0` (nibble 8) maps to bf16
      `0x8000` exactly as the table's `-0.0` entry.

    Parameters:
        width: SIMD width (lane count) of the nibble vector.

    Args:
        nibble: One E2M1 nibble per lane in the low 4 bits (`0..15`).

    Returns:
        The decoded values as `SIMD[.bfloat16, width]`, bit-identical to
        casting `E2M1_TO_FLOAT32[nibble]` to bfloat16.
    """
    var e = (nibble >> 1) & 0x3  # bits 2:1 -> exponent class
    var m = nibble & 0x1  # bit 0    -> mantissa
    var sign = (nibble & 0x8) << 12  # bit 3 -> bf16 sign bit (15)

    # Normal case (E >= 1): exp field = E + 126, mantissa MSB = m.
    var normal_mag = ((e + 126) << 7) | (m << 6)
    # Subnormal class (E == 0): m -> 0x3F00 (+0.5) or 0x0000 (+0.0).
    var subnormal_mag = m * 0x3F00

    var is_subnormal = e.eq(type_of(e)(0))  # SIMD[bool, width] mask
    var mag = is_subnormal.select(subnormal_mag, normal_mag)
    return bitcast[.bfloat16](sign | mag)


@always_inline
def decode_e2m1_to_f16[
    width: SIMDLength, //
](nibble: SIMD[.uint16, width]) -> SIMD[.float16, width]:
    """Decodes E2M1 nibbles to float16 by exponent injection (Preston's trick).

    The float16 twin of `decode_e2m1_to_f32_inject`, but -- unlike the f32/bf16
    injection variants -- **correct on the Apple M5** (and any FTZ target). It
    injects the 3 magnitude bits `(e1 e0 m0)` at float16 bits 11:9 and the sign
    at bit 15, then renormalizes with a single `* 2^14` (a power of two, hence
    exact). The result is **bit-identical** to `E2M1_TO_FLOAT32[nibble]` cast to
    float16 for all 16 values (every `{+-0, +-0.5, ..., +-6}` is exactly
    representable in float16), so `decode_e2m1_to_f16(n).cast[float32]()` equals
    `decode_e2m1_to_f32(n)` bit-for-bit.

    Why it is M5-safe where the f32/bf16 inject is not: the `+-0.5` codes
    (`E == 0, m == 1`) route through the float16 subnormal `0x0200` / `0x8200`
    (value `2^-15`), which `* 2^14` renormalizes to `+-0.5`. The Apple M5
    flushes **f32/bf16** denormals to zero on arithmetic inputs (see
    `patterns/apple-m5-denormal-flush-to-zero`) -- which is why
    `decode_e2m1_to_f32_inject` decodes `+-0.5` to `+-0` there -- but it
    **preserves float16 subnormals**, so this f16 decode keeps `+-0.5` exact
    (verified on-device: all 16 codes match `E2M1_TO_FLOAT32`). Callers that
    need f32/bf16 cast the f16 result afterwards; the cast of the now-normal
    `+-0.5` is exact.

    It is also cheaper than `decode_e2m1_to_f32` (no `uint32` widen, no
    `select`, no `E == 0` compare) while staying bit-exact -- the reason to
    prefer it on the M5 dequant path.

    Parameters:
        width: SIMD width (lane count) of the nibble vector.

    Args:
        nibble: One E2M1 nibble per lane in the low 4 bits (`0..15`).

    Returns:
        The decoded values as `SIMD[.float16, width]`, bit-identical to
        casting `E2M1_TO_FLOAT32[nibble]` to float16.
    """
    # (e1 e0 m0) -> f16 bits 11:9 (exp low 2 bits + mantissa MSB); sign -> 15.
    var mag = (nibble & 0x7) << 9
    var sign = (nibble & 0x8) << 12
    comptime c2_14 = bitcast[.float16](UInt16(0x7400))  # 2^14
    return bitcast[.float16](sign | mag) * c2_14


@always_inline
def decode_e2m1_to_f32[
    width: SIMDLength, //
](nibble: SIMD[.uint16, width]) -> SIMD[.float32, width]:
    """Decodes E2M1 nibbles to float32 with branch-free bit arithmetic.

    The float32-native twin of `decode_e2m1_to_bf16`: it builds the float32 bit
    pattern directly instead of constructing bf16 and widening. All 16 E2M1
    values are exactly representable in float32, so the result is
    **bit-identical** to `E2M1_TO_FLOAT32[nibble]` (and therefore to
    `decode_e2m1_to_bf16(nibble).cast[float32]()`). Use it on the dequant path
    where the next step is a float32 scale multiply -- it removes the bf16->f32
    widen per element while keeping the bit-exact-vs-table contract.

    Construction (float32 layout `s | 8-bit exp (bias 127) | 23-bit mantissa`):
    let `E` = bits2:1 and `m` = bit0.
    - Normal (`E >= 1`): exponent field `E + 126`, mantissa MSB `m` (mantissa
      field `m << 22`).
    - Subnormal-class (`E == 0`): `0.5 * m`, i.e. float32 `0x3F000000` (`+0.5`)
      when `m == 1`, else `0x00000000`. Selected branch-free.
    - Sign bit (`s`, nibble bit 3) shifted into float32 bit 31 (`s << 28`).

    Parameters:
        width: SIMD width (lane count) of the nibble vector.

    Args:
        nibble: One E2M1 nibble per lane in the low 4 bits (`0..15`).

    Returns:
        The decoded values as `SIMD[.float32, width]`, bit-identical to
        indexing `E2M1_TO_FLOAT32[nibble]`.
    """
    var n = nibble.cast[.uint32]()
    var e = (n >> 1) & 0x3  # bits 2:1 -> exponent class
    var m = n & 0x1  # bit 0    -> mantissa
    var sign = (n & 0x8) << 28  # bit 3 -> float32 sign bit (31)

    # Normal case (E >= 1): exp field = E + 126, mantissa MSB = m.
    var normal_mag = ((e + 126) << 23) | (m << 22)
    # Subnormal class (E == 0): m -> 0x3F000000 (+0.5) or 0x00000000 (+0.0).
    var subnormal_mag = m * 0x3F000000

    var is_subnormal = e.eq(type_of(e)(0))  # SIMD[bool, width] mask
    var mag = is_subnormal.select(subnormal_mag, normal_mag)
    return bitcast[.float32](sign | mag)


@always_inline
def decode_e2m1_to_f32_inject[
    width: SIMDLength, //
](nibble: SIMD[.uint16, width]) -> SIMD[.float32, width]:
    """Decodes E2M1 nibbles to float32 by exponent injection (Preston's trick).

    A branch-free alternative to `decode_e2m1_to_f32` with NO `select`: inject
    the 3 value bits `(e1 e0 m0)` and the sign directly into a float32 bit
    pattern at a SHIFTED exponent position, then renormalize with a single
    `* 2^126` multiply (a power-of-two scale, hence exact). The result is
    **bit-identical** to `decode_e2m1_to_f32` / `E2M1_TO_FLOAT32[nibble]` for all
    16 values on a denormal-honoring target (verified host-side, including `+-0`
    and the signed values).

    !!! warning "Wrong on flush-to-zero (FTZ) GPUs, including Apple M5"
        This trick relies on the `+-0.5` E2M1 values (`E == 0, m == 1`)
        producing a **denormal** float32 intermediate (`0x00400000`) that
        survives until the `* 2^126` renormalizes it. On a GPU that flushes
        denormals to zero on arithmetic inputs -- **the Apple M5 does** -- that
        intermediate is zeroed and `+-0.5` decodes to `+-0.0` (verified
        on-device: nibbles 1 and 9 mismatch, every other value is exact). Use
        `decode_e2m1_to_f32` (which builds the normalized value directly, no
        denormal intermediate) on Apple GPU and any FTZ target. This function is
        retained only for non-FTZ targets and as documentation of why the
        injection trick does not port to M5.

    Construction (float32 `s | 8-bit exp | 23-bit mantissa`): place `(e1 e0 m0)`
    at bits 22..24 and the sign at bit 31, leaving a denormalized/small float
    whose significand encodes the value; `* 2^126` shifts those bits into the
    proper exponent range, reproducing `2^(E-1) * (1 + m/2)` for `E >= 1` and
    `0.5 * m` for `E == 0` exactly (the same value the table holds). The
    `E == 0, m == 1` case is the lone denormal intermediate (see the warning).

    NOTE on folding: `2^126` CANNOT be folded into the per-block FP8 scale --
    `scale * 2^126` overflows float32 for any `scale >= 4.0` (fp8_e4m3 scales
    routinely exceed this), which corrupts the result. The `* 2^126` must be
    applied to the injected value FIRST (where the product is <= 6.0, no
    overflow), then the caller multiplies by the block scale. So this is NOT a
    multiply saved over `decode_e2m1_to_f32` -- it trades the `select` + the
    `(E+126)<<23 | m<<22` assembly for an extra `* 2^126`; whether it is faster
    is a per-target measurement (on Apple M5 it is moot -- it is wrong, see the
    warning -- and the dequant cost there is the scale LOAD, not the arith).

    Parameters:
        width: SIMD width (lane count) of the nibble vector.

    Args:
        nibble: One E2M1 nibble per lane in the low 4 bits (`0..15`).

    Returns:
        The decoded values as `SIMD[.float32, width]`, bit-identical to
        `E2M1_TO_FLOAT32[nibble]`.
    """
    var n = nibble.cast[.uint32]()
    # Inject (e1 e0 m0) at bits 22..24, sign (bit 3) at bit 31.
    var inj = ((n & 0x7) << 22) | ((n & 0x8) << 28)
    comptime c2_126 = bitcast[.float32](UInt32(0x7E800000))
    return bitcast[.float32](inj) * c2_126


@always_inline
def compute_mxfp4_even_scale(max_val: Float32) -> Float8_e8m0fnu:
    """Computes the OCP MXFP4 E8M0 scale using even-mode rounding.

    Even-mode rounding rounds the block maximum before deriving the scale
    exponent. This differs from ceil(max / 6) and preserves more precision for
    smaller values in the same 32-element block.
    """
    comptime FP32_MANTISSA_WIDTH = FPUtils[.float32].mantissa_width()
    # MXFP4 stores only a power-of-two scale. Pick the scale so the largest
    # value in the block still fits in FP4 E2M1 after rounding, where the
    # largest finite FP4 E2M1 value is 6.0 = 1.5 * 2^2.
    #
    # The add below rounds max_val at the FP4 mantissa boundary. If that
    # rounded value crosses into the next power-of-two bucket, its Float32
    # exponent increases. Subtracting 2 (the exponent of FP4's max value)
    # turns that rounded-max exponent into the E8M0 scale exponent.
    #
    # Conceptually, for a block like [1.6, 0.4, ...], even-mode chooses
    # scale 0.25: [1.6, 0.4] / 0.25 = [6.4, 1.6], which rounds to FP4
    # [6.0, 1.5] and dequantizes to [1.5, 0.375]. Ceil(max / 6) would choose
    # scale 0.5: [1.6, 0.4] / 0.5 = [3.2, 0.8], which rounds to FP4
    # [3.0, 1.0] and dequantizes to [1.5, 0.5].
    comptime ROUND_TO_FP4_E2M1_MANTISSA = 1 << (
        FP32_MANTISSA_WIDTH - FP4_E2M1_MANTISSA_WIDTH - 1
    )
    var max_bits = FPUtils[.float32].bitcast_to_uint(max_val)
    var rounded_max_bits = max_bits + type_of(max_bits)(
        ROUND_TO_FP4_E2M1_MANTISSA
    )
    var rounded_max = bitcast[.float32](rounded_max_bits)
    var scale_exp = (
        FPUtils[.float32].get_exponent_biased(rounded_max)
        - FP4_E2M1_MAX_EXPONENT
    )
    scale_exp = max(0, min(scale_exp, 254))
    return bitcast[.float8_e8m0fnu](UInt8(scale_exp))


def cast_uint_to_fp4e2m1[
    in_dtype: DType,
    in_width: SIMDLength,
    //,
    *,
    out_dtype: DType,
    out_width: Int,
](x: SIMD[in_dtype, in_width]) -> SIMD[out_dtype, out_width]:
    """Unpacks FP4 E2M1 nibbles packed inside unsigned integers to floating-point values.

    Each input integer (uint8, uint16, or uint32) holds multiple packed 4-bit E2M1 values.
    The function extracts each nibble, looks up its float32 value in `E2M1_TO_FLOAT32`,
    casts to `out_dtype`, and assembles the result vector.

    Parameters:
        in_dtype: Unsigned integer type holding packed FP4 nibbles (uint8, uint16, or uint32).
        in_width: Number of input integer elements.
        out_dtype: Output floating-point element type.
        out_width: Total output elements; must equal `in_width * (bit_width(in_dtype) // 4)`.

    Args:
        x: Packed FP4 E2M1 input vector.

    Returns:
        Decoded floating-point vector of length out_width.
    """
    comptime assert in_dtype in (
        DType.uint32,
        DType.uint16,
        DType.uint8,
    ), "input_dtype must be uint32, uint16 or uint8"

    comptime FP4_E2M1_WIDTH = 4
    comptime FP4_E2M1_MASK = pow(2, FP4_E2M1_WIDTH) - 1
    comptime num_fp4_values = bit_width_of[in_dtype]() // FP4_E2M1_WIDTH

    comptime assert in_width * num_fp4_values == out_width, (
        "size mismatch: input_width * num_fp4_values must be equal to"
        " output_width"
    )

    var result = SIMD[out_dtype, out_width]()

    comptime for i in range(in_width):
        comptime for shift in range(0, num_fp4_values):
            comptime BitsType = type_of(x[i].to_bits())
            var x = (
                x[i].to_bits() >> BitsType(shift * FP4_E2M1_WIDTH)
            ) & BitsType(FP4_E2M1_MASK)
            result[i * num_fp4_values + shift] = E2M1_TO_FLOAT32[Int(x)].cast[
                out_dtype
            ]()
    return result


def cast_fp_to_fp4e2m1[
    dtype: DType,
    width: SIMDLength,
    //,
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Rounds each floating-point element to the nearest FP4 E2M1 representable value.

    Implements nearest-even rounding into the 16 representable FP4 E2M1 values:
    {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. Values outside [−6, 6] saturate to ±6.
    The result is returned in the original dtype, not packed as nibbles.

    Parameters:
        dtype: Floating-point element type (float32, bfloat16, or float16).
        width: SIMD lane count.

    Args:
        x: Input floating-point vector.

    Returns:
        Vector of the same dtype with each element rounded to the nearest FP4 E2M1 value.
    """
    comptime assert dtype in (
        DType.float32,
        DType.bfloat16,
        DType.float16,
    ), "dtype must be float32, bfloat16 or float16"
    # for float4_e2m1fn has only 16 values
    # (x >= 0.0) & (x <= 0.25)] => 0.0
    # (x > 0.25) & (x < 0.75)] => 0.5
    # (x >= 0.75) & (x <= 1.25)] => 1.0
    # (x > 1.25) & (x < 1.75)] => 1.5
    # (x >= 1.75) & (x <= 2.5)] => 2.0
    # (x > 2.5) & (x < 3.5)] => 3.0
    # (x >= 3.5) & (x <= 5.0)] => 4.0
    # (x > 5.0) => 6.0

    var sign = x.lt(0).select(-1.0, 1.0).cast[dtype]()
    var abs_x = abs(x)
    var result = SIMD[dtype, width]()

    comptime for i in range(width):
        if abs_x[i] <= 0.25:
            result[i] = 0.0
        elif abs_x[i] < 0.75:
            result[i] = 0.5
        elif abs_x[i] <= 1.25:
            result[i] = 1.0
        elif abs_x[i] < 1.75:
            result[i] = 1.5
        elif abs_x[i] <= 2.5:
            result[i] = 2.0
        elif abs_x[i] < 3.5:
            result[i] = 3.0
        elif abs_x[i] <= 5.0:
            result[i] = 4.0
        else:
            result[i] = 6.0
    return result * sign


def cast_fp32_to_fp4e2m1[
    width: SIMDLength,
    //,
](x: SIMD[.float32, width]) -> UInt32:
    """Converts eight float32 values to a packed FP4 E2M1 word using SM100 PTX instructions.

    Issues four `cvt.rn.satfinite.e2m1x2.f32` PTX instructions to convert pairs of
    float32 values to 4-bit FP4 E2M1, packing the results into a single UInt32.

    Parameters:
        width: Must be 8; each call converts exactly eight float32 values.

    Args:
        x: Eight float32 input values to convert.

    Returns:
        A UInt32 with the eight FP4 E2M1 nibbles packed in byte order.

    Constraints:
        Requires NVIDIA GPU with SM100 or newer (B200 and above).
    """
    comptime assert (
        is_nvidia_gpu() and _is_sm_100x_or_newer()
    ), "only supported on NVIDIA GPUs with SM 100 or newer"
    comptime assert width == 8, "width must be 8"

    comptime asm_code = """{
.reg .b8 byte0;
.reg .b8 byte1;
.reg .b8 byte2;
.reg .b8 byte3;
cvt.rn.satfinite.e2m1x2.f32   byte0, $2, $1;
cvt.rn.satfinite.e2m1x2.f32   byte1, $4, $3;
cvt.rn.satfinite.e2m1x2.f32   byte2, $6, $5;
cvt.rn.satfinite.e2m1x2.f32   byte3, $8, $7;
mov.b32 $0, {byte0, byte1, byte2, byte3};
}
"""
    return inlined_assembly[
        asm_code, UInt32, constraints="=r,f,f,f,f,f,f,f,f", has_side_effect=True
    ](x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])


def cast_f4e2m1x2_to_fp16x2(x: UInt8) -> SIMD[.float16, 2]:
    """Converts two FP4 E2M1 nibbles packed in one byte to two float16 values using SM100 PTX.

    Issues the `cvt.rn.f16x2.e2m1x2` PTX instruction to decode both nibbles in a
    single hardware operation.

    Args:
        x: A uint8 containing two packed FP4 E2M1 nibbles (low nibble = element 0).

    Returns:
        A two-element float16 SIMD vector with the decoded values.

    Constraints:
        Requires NVIDIA GPU with SM100 or newer (B200 and above).
    """
    comptime assert (
        is_nvidia_gpu() and _is_sm_100x_or_newer()
    ), "only supported on NVIDIA GPUs with SM 100 or newer"

    comptime asm_code = """{
.reg .b8 byte0;
.reg .b8 byte1;
mov.b16 {byte0, byte1}, $1;
cvt.rn.f16x2.e2m1x2 $0, byte0;
}
"""
    var result = inlined_assembly[
        asm_code, UInt32, constraints="=r,h", has_side_effect=True
    ](UInt16(x))

    return bitcast[.float16, 2](result)


@always_inline
def cast_float_to_fp4e2m1_amd[
    dtype: DType, width: SIMDLength, //
](input: SIMD[dtype, width], scale: Float32) -> UInt32:
    """Converts up to eight floating-point values to packed FP4 E2M1 using AMD CDNA4 intrinsics.

    Applies the provided scale factor before quantization using the
    `llvm.amdgcn.cvt.scalef32.pk.fp4.*` intrinsics, packing pairs of values per call.

    Parameters:
        dtype: Input element type (bfloat16 or float32).
        width: Number of input elements; must be even and at most 8.

    Args:
        input: Input floating-point vector to convert.
        scale: Scale factor applied before FP4 quantization.

    Returns:
        A UInt32 with the converted FP4 E2M1 nibbles packed in order.

    Constraints:
        Requires AMD CDNA4 or newer (MI355X and above).
    """
    comptime assert (
        _cdna_4_or_newer()
    ), "only supported on AMD CDNA4 or newer (MI355X)"
    comptime assert (
        width % 2 == 0 and width <= 8
    ), "width must be even and at most 8"

    var packed = UInt32(0)

    comptime for i in range(width // 2):
        comptime if dtype == .bfloat16:
            packed = llvm_intrinsic[
                "llvm.amdgcn.cvt.scalef32.pk.fp4.bf16",
                UInt32,
            ](packed, input.slice[2, offset=i * 2](), scale, Int32(i))
        elif dtype == .float32:
            packed = llvm_intrinsic[
                "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
                UInt32,
            ](packed, input[i * 2], input[i * 2 + 1], scale, Int32(i))
        else:
            comptime assert False, "Unsupported dtype"

    return packed


def set_scale_factor[
    scales_dtype: DType,
    scales_layout: Layout,
    //,
    SF_VECTOR_SIZE: Int,
    width: SIMDLength,
](
    scales_tensor: LayoutTensor[mut=True, scales_dtype, scales_layout, ...],
    row_idx: Int,
    col_idx: Int,
    scale_value: SIMD[scales_dtype, width],
):
    """Stores a scale factor into a 5D non-batched `LayoutTensor` at the given row and column.

    Translates the linear `(row_idx, col_idx)` coordinates into the 5D
    scale-factor layout used by MXFP4/NVFP4 block-scaled tensors and stores
    `scale_value` there with natural alignment.

    Parameters:
        scales_dtype: Element type of the scales tensor.
        scales_layout: Layout of the scales `LayoutTensor`.
        SF_VECTOR_SIZE: Number of elements each scale factor covers.
        width: SIMD width of the value to store; must not exceed `SF_ATOM_K`.

    Args:
        scales_tensor: Mutable 5D `LayoutTensor` holding the scale factors.
        row_idx: Row index in the original (unscaled) tensor coordinates.
        col_idx: Column index in the original (unscaled) tensor coordinates.
        scale_value: Scale factor value(s) to store.
    """
    comptime assert (
        scales_tensor.rank == 5
    ), "scales_tensor must be 5D for non-batched scales tensor"
    comptime assert (
        width <= SF_ATOM_K
    ), "width must be less than or equal to SF_ATOM_K"

    comptime align = align_of[SIMD[scales_dtype, width]]()
    scales_tensor.store[store_alignment=align](
        IndexList[5](
            row_idx // SF_MN_GROUP_SIZE,
            col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
            row_idx % SF_ATOM_M[0],
            (row_idx % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
            (col_idx // SF_VECTOR_SIZE) % SF_ATOM_K,
        ),
        scale_value,
    )


def set_scale_factor[
    scales_dtype: DType,
    width: SIMDLength,
    //,
    SF_VECTOR_SIZE: Int,
](
    scales_tensor: TileTensor[mut=True, scales_dtype, ...],
    row_idx: Int,
    col_idx: Int,
    scale_value: SIMD[scales_dtype, width],
):
    """Stores a scale factor into a 5D non-batched `TileTensor` at the given row and column.

    TileTensor overload of `set_scale_factor` that translates the linear
    `(row_idx, col_idx)` coordinates into the 5D scale-factor layout and
    stores `scale_value` there.

    Parameters:
        scales_dtype: Element type of the scales tensor.
        width: SIMD width of the value to store; must not exceed `SF_ATOM_K`.
        SF_VECTOR_SIZE: Number of elements each scale factor covers.

    Args:
        scales_tensor: Mutable `TileTensor` (flat rank >= 5) holding the scale factors.
        row_idx: Row index in the original (unscaled) tensor coordinates.
        col_idx: Column index in the original (unscaled) tensor coordinates.
        scale_value: Scale factor value(s) to store.
    """
    comptime assert (
        width <= SF_ATOM_K
    ), "width must be less than or equal to SF_ATOM_K"
    comptime assert scales_tensor.flat_rank >= 5, "scales_tensor must be 5D"

    scales_tensor.store[width=width](
        (
            row_idx // SF_MN_GROUP_SIZE,
            col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
            row_idx % SF_ATOM_M[0],
            (row_idx % SF_MN_GROUP_SIZE // SF_ATOM_M[0]),
            (col_idx // SF_VECTOR_SIZE % SF_ATOM_K),
        ),
        scale_value,
    )


def get_scale_factor[
    scales_dtype: DType,
    scales_layout: Layout,
    //,
    SF_VECTOR_SIZE: Int,
](
    scales_tensor: LayoutTensor[scales_dtype, scales_layout, MutAnyOrigin],
    row_idx: Int,
    col_idx: Int,
) -> Scalar[scales_dtype]:
    """Loads a scale factor from a 5D non-batched `LayoutTensor` at the given row and column.

    Translates the linear `(row_idx, col_idx)` coordinates into the 5D
    scale-factor layout used by MXFP4/NVFP4 block-scaled tensors and returns
    the stored scale factor.

    Parameters:
        scales_dtype: Element type of the scales tensor.
        scales_layout: Layout of the scales `LayoutTensor`.
        SF_VECTOR_SIZE: Number of elements each scale factor covers.

    Args:
        scales_tensor: 5D `LayoutTensor` holding the scale factors.
        row_idx: Row index in the original (unscaled) tensor coordinates.
        col_idx: Column index in the original (unscaled) tensor coordinates.

    Returns:
        The scale factor stored at the translated 5D coordinate.
    """
    comptime assert (
        scales_tensor.rank == 5
    ), "scales_tensor must be 5D for non-batched scales tensor"

    return rebind[Scalar[scales_dtype]](
        scales_tensor[
            row_idx // SF_MN_GROUP_SIZE,
            col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
            row_idx % SF_ATOM_M[0],
            (row_idx % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
            (col_idx // SF_VECTOR_SIZE) % SF_ATOM_K,
        ]
    )


def get_scale_factor[
    scales_dtype: DType,
    //,
    SF_VECTOR_SIZE: Int,
](
    scales_tensor: TileTensor[mut=True, scales_dtype, ...],
    row_idx: Int,
    col_idx: Int,
) -> Scalar[scales_dtype]:
    """Loads a scale factor from a 5D non-batched `TileTensor` at the given row and column.

    TileTensor overload of `get_scale_factor` that translates the linear
    `(row_idx, col_idx)` coordinates into the 5D scale-factor layout and
    returns the stored scale factor.

    Parameters:
        scales_dtype: Element type of the scales tensor.
        SF_VECTOR_SIZE: Number of elements each scale factor covers.

    Args:
        scales_tensor: `TileTensor` (flat rank >= 5) holding the scale factors.
        row_idx: Row index in the original (unscaled) tensor coordinates.
        col_idx: Column index in the original (unscaled) tensor coordinates.

    Returns:
        The scale factor stored at the translated 5D coordinate.
    """
    comptime assert (
        scales_tensor.flat_rank >= 5
    ), "scales_tensor must be 5D for non-batched scales tensor"

    return rebind[Scalar[scales_dtype]](
        scales_tensor[
            Coord(
                row_idx // SF_MN_GROUP_SIZE,
                col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
                row_idx % SF_ATOM_M[0],
                (row_idx % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
                (col_idx // SF_VECTOR_SIZE) % SF_ATOM_K,
            )
        ]
    )


def set_batched_scale_factor[
    scales_dtype: DType,
    scales_layout: Layout,
    //,
    SF_VECTOR_SIZE: Int,
](
    scales_tensor: LayoutTensor[scales_dtype, scales_layout, MutAnyOrigin],
    batch_idx: Int,
    row_idx: Int,
    col_idx: Int,
    scale_value: Scalar[scales_dtype],
):
    """Stores a scale factor into a 6D batched `LayoutTensor` at the given batch, row, and column.

    Translates the linear `(batch_idx, row_idx, col_idx)` coordinates into the
    6D scale-factor layout used by batched MXFP4/NVFP4 block-scaled tensors and
    stores `scale_value` there.

    Parameters:
        scales_dtype: Element type of the scales tensor.
        scales_layout: Layout of the scales `LayoutTensor`.
        SF_VECTOR_SIZE: Number of elements each scale factor covers.

    Args:
        scales_tensor: Mutable 6D `LayoutTensor` holding the batched scale factors.
        batch_idx: Batch index in the original (unscaled) tensor coordinates.
        row_idx: Row index in the original (unscaled) tensor coordinates.
        col_idx: Column index in the original (unscaled) tensor coordinates.
        scale_value: Scale factor value to store.
    """
    comptime assert (
        scales_tensor.rank == 6
    ), "scales_tensor must be 6D for batched scales tensor"

    scales_tensor[
        batch_idx,
        row_idx // SF_MN_GROUP_SIZE,
        col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
        row_idx % SF_ATOM_M[0],
        (row_idx % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
        (col_idx // SF_VECTOR_SIZE) % SF_ATOM_K,
    ] = scale_value


def set_batched_scale_factor[
    scales_dtype: DType,
    //,
    SF_VECTOR_SIZE: Int,
](
    scales_tensor: TileTensor[mut=True, scales_dtype, ...],
    batch_idx: Int,
    row_idx: Int,
    col_idx: Int,
    scale_value: Scalar[scales_dtype],
):
    """Stores a scale factor into a 6D batched `TileTensor` at the given batch, row, and column.

    TileTensor overload of `set_batched_scale_factor` that translates the
    linear `(batch_idx, row_idx, col_idx)` coordinates into the 6D scale-factor
    layout and stores `scale_value` there.

    Parameters:
        scales_dtype: Element type of the scales tensor.
        SF_VECTOR_SIZE: Number of elements each scale factor covers.

    Args:
        scales_tensor: Mutable `TileTensor` (flat rank == 6) holding the batched scale factors.
        batch_idx: Batch index in the original (unscaled) tensor coordinates.
        row_idx: Row index in the original (unscaled) tensor coordinates.
        col_idx: Column index in the original (unscaled) tensor coordinates.
        scale_value: Scale factor value to store.
    """
    comptime assert (
        scales_tensor.flat_rank == 6
    ), "scales_tensor must be 6D for batched scales tensor"

    scales_tensor.store(
        (
            batch_idx,
            row_idx // SF_MN_GROUP_SIZE,
            col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
            row_idx % SF_ATOM_M[0],
            (row_idx % SF_MN_GROUP_SIZE // SF_ATOM_M[0]),
            (col_idx // SF_VECTOR_SIZE % SF_ATOM_K),
        ),
        scale_value,
    )


def get_batched_scale_factor[
    scales_dtype: DType,
    scales_layout: Layout,
    //,
    SF_VECTOR_SIZE: Int,
](
    scales_tensor: LayoutTensor[scales_dtype, scales_layout, MutAnyOrigin],
    batch_idx: Int,
    row_idx: Int,
    col_idx: Int,
) -> Scalar[scales_dtype]:
    """Loads a scale factor from a 6D batched `LayoutTensor` at the given batch, row, and column.

    Translates the linear `(batch_idx, row_idx, col_idx)` coordinates into the
    6D scale-factor layout used by batched MXFP4/NVFP4 block-scaled tensors and
    returns the stored scale factor.

    Parameters:
        scales_dtype: Element type of the scales tensor.
        scales_layout: Layout of the scales `LayoutTensor`.
        SF_VECTOR_SIZE: Number of elements each scale factor covers.

    Args:
        scales_tensor: 6D `LayoutTensor` holding the batched scale factors.
        batch_idx: Batch index in the original (unscaled) tensor coordinates.
        row_idx: Row index in the original (unscaled) tensor coordinates.
        col_idx: Column index in the original (unscaled) tensor coordinates.

    Returns:
        The scale factor stored at the translated 6D coordinate.
    """
    comptime assert (
        scales_tensor.rank == 6
    ), "scales_tensor must be 6D for batched scales tensor"

    return rebind[Scalar[scales_dtype]](
        scales_tensor[
            batch_idx,
            row_idx // SF_MN_GROUP_SIZE,
            col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
            row_idx % SF_ATOM_M[0],
            (row_idx % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
            (col_idx // SF_VECTOR_SIZE) % SF_ATOM_K,
        ]
    )


def get_batched_scale_factor[
    scales_dtype: DType,
    //,
    SF_VECTOR_SIZE: Int,
](
    scales_tensor: TileTensor[mut=True, scales_dtype, ...],
    batch_idx: Int,
    row_idx: Int,
    col_idx: Int,
) -> Scalar[scales_dtype]:
    """Loads a scale factor from a 6D batched `TileTensor` at the given batch, row, and column.

    TileTensor overload of `get_batched_scale_factor` that translates the
    linear `(batch_idx, row_idx, col_idx)` coordinates into the 6D scale-factor
    layout and returns the stored scale factor.

    Parameters:
        scales_dtype: Element type of the scales tensor.
        SF_VECTOR_SIZE: Number of elements each scale factor covers.

    Args:
        scales_tensor: `TileTensor` (flat rank == 6) holding the batched scale factors.
        batch_idx: Batch index in the original (unscaled) tensor coordinates.
        row_idx: Row index in the original (unscaled) tensor coordinates.
        col_idx: Column index in the original (unscaled) tensor coordinates.

    Returns:
        The scale factor stored at the translated 6D coordinate.
    """
    comptime assert (
        scales_tensor.flat_rank == 6
    ), "scales_tensor must be 6D for batched scales tensor"

    return rebind[Scalar[scales_dtype]](
        scales_tensor[
            Coord(
                batch_idx,
                row_idx // SF_MN_GROUP_SIZE,
                col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
                row_idx % SF_ATOM_M[0],
                (row_idx % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
                (col_idx // SF_VECTOR_SIZE) % SF_ATOM_K,
            )
        ]
    )


def convert_ref_scales_to_mxfp8_format[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    ref_scales_type: DType,
    scales_type: DType,
    ref_a_scales_layout: Layout,
    ref_b_scales_layout: Layout,
    a_scales_layout: Layout,
    b_scales_layout: Layout,
    a_scales_origin: MutOrigin,
    b_scales_origin: MutOrigin,
    *,
    REF_BLOCK_SIZE: Int,
    SF_VECTOR_SIZE: Int,
](
    m: MType,
    n: NType,
    k: KType,
    ref_a_scales: LayoutTensor[ref_scales_type, ref_a_scales_layout, _],
    ref_b_scales: LayoutTensor[ref_scales_type, ref_b_scales_layout, _],
    a_scales: LayoutTensor[scales_type, a_scales_layout, a_scales_origin],
    b_scales: LayoutTensor[scales_type, b_scales_layout, b_scales_origin],
):
    """Converts reference float32 block scales into the 5D MXFP8 E8M0 scale-factor layout.

    Reads the per-block float32 reference scales for the A (M x K) and
    B (N x K) operands, converts each to `float8_e8m0fnu`, and writes them into
    the corresponding 5D scale-factor tensors expected by block-scaled matmul
    kernels.

    Parameters:
        MType: CoordLike type carrying the M dimension size.
        NType: CoordLike type carrying the N dimension size.
        KType: CoordLike type carrying the K dimension size.
        ref_scales_type: Element type of the reference scales (must be float32).
        scales_type: Element type of the output scales (must be float8_e8m0fnu).
        ref_a_scales_layout: Layout of the 2D reference A scales tensor.
        ref_b_scales_layout: Layout of the 2D reference B scales tensor.
        a_scales_layout: Layout of the 5D output A scales tensor.
        b_scales_layout: Layout of the 5D output B scales tensor.
        a_scales_origin: Mutability origin of the output A scales tensor.
        b_scales_origin: Mutability origin of the output B scales tensor.
        REF_BLOCK_SIZE: Block size (in elements) used by the reference scales.
        SF_VECTOR_SIZE: Number of elements each scale factor covers in the output layout.

    Args:
        m: M dimension size of the operands.
        n: N dimension size of the operands.
        k: K dimension size of the operands.
        ref_a_scales: 2D float32 reference scales for the A operand, indexed as `[k // REF_BLOCK_SIZE, m]`.
        ref_b_scales: 2D float32 reference scales for the B operand, indexed as `[n // REF_BLOCK_SIZE, k // REF_BLOCK_SIZE]`.
        a_scales: Mutable 5D output tensor receiving the converted A scales.
        b_scales: Mutable 5D output tensor receiving the converted B scales.
    """
    comptime assert (
        ref_scales_type == .float32
    ), "Only support float32 reference scales"
    comptime assert (
        scales_type == .float8_e8m0fnu
    ), "Only support float8_e8m0fnu scales"
    comptime assert ref_a_scales_layout.rank() == 2, "ref_a_scales must be 2D"
    comptime assert ref_b_scales_layout.rank() == 2, "ref_b_scales must be 2D"
    comptime assert a_scales_layout.rank() == 5, "a_scales must be 5D"
    comptime assert b_scales_layout.rank() == 5, "b_scales must be 5D"

    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    # initialize a_scales_tensor and b_scales_tensor based on reference scales
    for m in range(M):
        for k in range(K):
            a_scales[
                m // SF_MN_GROUP_SIZE,
                k // (SF_VECTOR_SIZE * SF_ATOM_K),
                m % SF_ATOM_M[0],
                (m % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
                k % SF_ATOM_K,
            ] = rebind[Scalar[scales_type]](
                _convert_f32_to_float8_ue8m0[scales_type](
                    ref_a_scales[k // REF_BLOCK_SIZE, m]
                )
            )

    for n in range(N):
        for k in range(K):
            b_scales[
                n // SF_MN_GROUP_SIZE,
                k // (SF_VECTOR_SIZE * SF_ATOM_K),
                n % SF_ATOM_M[0],
                (n % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
                k % SF_ATOM_K,
            ] = rebind[Scalar[scales_type]](
                _convert_f32_to_float8_ue8m0[scales_type](
                    ref_b_scales[n // REF_BLOCK_SIZE, k // REF_BLOCK_SIZE]
                )
            )


def block_scaled_umma_kind[
    a_type: DType, b_type: DType, scales_dtype: DType
]() -> UMMAKind:
    """Selects the SM100 UMMA kind matching the operand and scale-factor types.

    The single mapping from operand dtypes to `UMMAKind`; callers that also know
    the scale-factor vector size should go through `get_scaling_kind`.

    B's dtype has to be read alongside A's: W4A8 and plain MXFP8 share A's E4M3
    and the E8M0 scales, and differ only in B. Returning `kind::mxf8f6f4` for
    the mixed pair only says the tensor cores accept it -- the caller still owes
    the padded FP4 TMA copy that feeds B (see `_is_packed_fp4`), so a path
    without one must reject the pair rather than call this.

    Parameters:
        a_type: A operand element type; `uint8` is nibble-packed E2M1.
        b_type: B operand element type; `uint8` is nibble-packed E2M1.
        scales_dtype: Scale-factor element type.

    Returns:
        The `UMMAKind` matching the provided type combination.
    """
    comptime if is_w4a8_operand_pair[a_type, b_type]():
        comptime assert (
            scales_dtype == MXFP8_SF_DTYPE
        ), "W4A8 requires E8M0 scales on both operands"
        return UMMAKind.KIND_MXF8F6F4
    elif a_type == .uint8 and scales_dtype == NVFP4_SF_DTYPE:
        return UMMAKind.KIND_MXF4NVF4
    elif a_type == .uint8 and scales_dtype == MXFP4_SF_DTYPE:
        return UMMAKind.KIND_MXF4
    else:
        comptime assert (
            a_type == .float8_e4m3fn and scales_dtype == MXFP8_SF_DTYPE
        ), "unsupported a_type/scales_dtype for block-scaled matmul"
        comptime assert (
            a_type == b_type
        ), "mixed operand dtypes are only supported for the W4A8 pair"
        return UMMAKind.KIND_MXF8F6F4


def get_scaling_kind[
    a_type: DType,
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
    b_type: DType,
]() -> UMMAKind:
    """Selects the UMMA kind for a same-format block-scaled matmul.

    `block_scaled_umma_kind` plus the scale-factor vector-size check. Rejects
    the mixed W4A8 pair: only the grouped 1D-1D kernel implements the padded FP4
    TMA copy that mixed operands need, and it infers its kind directly.

    Parameters:
        a_type: A operand element type (`uint8` for MXFP4/NVFP4,
            `float8_e4m3fn` for MXFP8).
        scales_dtype: Scale-factor element type.
        SF_VECTOR_SIZE: Number of elements each scale factor covers.
        b_type: B operand element type; must equal `a_type`.

    Returns:
        The `UMMAKind` matching the provided type combination.
    """
    comptime assert a_type == b_type, (
        "this path requires both operands in the same format; the mixed W4A8"
        " pair is only supported by the grouped block-scaled matmul"
    )
    # The scale dtype fixes the group size: NVFP4's E4M3 scales cover 16
    # elements, and MXFP4 and MXFP8 share E8M0 scales over 32.
    comptime expected_sf_vector_size = (
        NVFP4_SF_VECTOR_SIZE if scales_dtype
        == NVFP4_SF_DTYPE else MXFP4_SF_VECTOR_SIZE
    )
    comptime assert (
        SF_VECTOR_SIZE == expected_sf_vector_size
    ), "SF_VECTOR_SIZE does not match the scale dtype's block-scaled format"
    return block_scaled_umma_kind[a_type, b_type, scales_dtype]()
