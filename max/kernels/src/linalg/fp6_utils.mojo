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

"""Provides low-level FP6 encode/decode utilities and scale-factor helpers.

The FP6 twin of `fp4_utils`, covering both OCP MX FP6 encodings (E2M3 and
E3M2). Neither FP6 encoding has Inf or NaN: all 64 codes are finite, so decode
needs no special-case branch and encode saturates rather than propagating a
non-finite input.

Both encodings share the MX block structure with MXFP4 -- 32 consecutive
elements along K share one E8M0 scale -- so `MXFP6_SF_VECTOR_SIZE` and
`MXFP6_SF_DTYPE` match their MXFP4 counterparts exactly.
"""

from std.os import abort
from std.utils.numerics import FPUtils
from std.memory import bitcast

comptime MXFP6_SF_VECTOR_SIZE = 32
comptime MXFP6_SF_DTYPE = DType.float8_e8m0fnu

comptime FP6_ELEMENTS_PER_GROUP = 4
comptime FP6_BYTES_PER_GROUP = 3


@fieldwise_init
struct FP6Format(Equatable, TrivialRegisterPassable):
    """Selects between the two OCP MX FP6 element encodings.

    The numeric parameters returned by the accessors below are the entire
    difference between the two formats; every routine in this module is
    parameterized on this type rather than duplicated per encoding.
    """

    var _value: Int32

    comptime E2M3 = Self(0)
    comptime E3M2 = Self(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def mantissa_width(self) -> Int:
        """Returns the mantissa field width in bits."""
        if self == FP6Format.E2M3:
            return 3
        if self == FP6Format.E3M2:
            return 2
        abort("invalid FP6 format")

    def exponent_width(self) -> Int:
        """Returns the exponent field width in bits."""
        if self == FP6Format.E2M3:
            return 2
        if self == FP6Format.E3M2:
            return 3
        abort("invalid FP6 format")

    def exponent_bias(self) -> Int:
        """Returns the exponent bias."""
        if self == FP6Format.E2M3:
            return 1
        if self == FP6Format.E3M2:
            return 3
        abort("invalid FP6 format")

    def max_exponent(self) -> Int:
        """Returns the unbiased exponent of the largest finite value.

        E2M3 tops out at `7.5 == 1.875 * 2^2` and E3M2 at `28 == 1.75 * 2^4`,
        so this is 2 and 4 respectively. `compute_mxfp6_even_scale` subtracts
        it from the block maximum's exponent to derive the E8M0 scale.
        """
        if self == FP6Format.E2M3:
            return 2
        if self == FP6Format.E3M2:
            return 4
        abort("invalid FP6 format")

    def max_value(self) -> Float32:
        """Returns the largest finite magnitude the encoding represents."""
        if self == FP6Format.E2M3:
            return 7.5
        if self == FP6Format.E3M2:
            return 28.0
        abort("invalid FP6 format")


comptime E2M3_TO_FLOAT32 = SIMD[.float32, 64](
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
    1.125,
    1.25,
    1.375,
    1.5,
    1.625,
    1.75,
    1.875,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    -0.0,
    -0.125,
    -0.25,
    -0.375,
    -0.5,
    -0.625,
    -0.75,
    -0.875,
    -1.0,
    -1.125,
    -1.25,
    -1.375,
    -1.5,
    -1.625,
    -1.75,
    -1.875,
    -2.0,
    -2.25,
    -2.5,
    -2.75,
    -3.0,
    -3.25,
    -3.5,
    -3.75,
    -4.0,
    -4.5,
    -5.0,
    -5.5,
    -6.0,
    -6.5,
    -7.0,
    -7.5,
)

comptime E3M2_TO_FLOAT32 = SIMD[.float32, 64](
    0.0,
    0.0625,
    0.125,
    0.1875,
    0.25,
    0.3125,
    0.375,
    0.4375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    10.0,
    12.0,
    14.0,
    16.0,
    20.0,
    24.0,
    28.0,
    -0.0,
    -0.0625,
    -0.125,
    -0.1875,
    -0.25,
    -0.3125,
    -0.375,
    -0.4375,
    -0.5,
    -0.625,
    -0.75,
    -0.875,
    -1.0,
    -1.25,
    -1.5,
    -1.75,
    -2.0,
    -2.5,
    -3.0,
    -3.5,
    -4.0,
    -5.0,
    -6.0,
    -7.0,
    -8.0,
    -10.0,
    -12.0,
    -14.0,
    -16.0,
    -20.0,
    -24.0,
    -28.0,
)


@always_inline
def fp6_reference_table[fmt: FP6Format]() -> SIMD[.float32, 64]:
    """Returns the hand-written decode table for `fmt`.

    Parameters:
        fmt: The FP6 encoding to look up.

    Returns:
        All 64 decoded values, indexed by raw 6-bit code.
    """
    comptime if fmt == FP6Format.E2M3:
        return E2M3_TO_FLOAT32
    else:
        return E3M2_TO_FLOAT32


@always_inline
def decode_fp6_to_f32[
    width: SIMDLength, //, fmt: FP6Format
](code: SIMD[.uint8, width]) -> SIMD[.float32, width]:
    """Decodes FP6 codes to float32 with branch-free bit arithmetic.

    Builds the float32 bit pattern directly, so the result is **bit-identical**
    to indexing `fp6_reference_table[fmt]()` for all 64 codes, including the
    signed zeros. Every FP6 value is exactly representable in float32.

    Construction (float32 `s | 8-bit exp (bias 127) | 23-bit mantissa`), with
    `E` the exponent field, `m` the mantissa field, and `M` the mantissa width:

    - Normal (`E >= 1`): value `2^(E - bias) * (1 + m/2^M)`, so the float32
      exponent field is `E + 127 - bias` and the mantissa field is
      `m << (23 - M)`.
    - Subnormal (`E == 0`): value `m * 2^(1 - bias - M)`, a uniform grid. Built
      by converting `m` to float32 (exact for `m < 2^M`) and scaling by a power
      of two (also exact) rather than by bit surgery, which would need a
      leading-zero count.
    - The sign bit (code bit 5) is OR'd into float32 bit 31 after the select,
      so `-0` decodes to `0x80000000` and matches the table's `-0.0`.

    Unlike `decode_e2m1_to_f32_inject` in `fp4_utils`, this never forms a
    denormal float32 intermediate, so it is correct on flush-to-zero targets.

    Parameters:
        width: SIMD width (lane count) of the code vector.
        fmt: The FP6 encoding of the input codes.

    Args:
        code: One FP6 code per lane in the low 6 bits (`0..63`).

    Returns:
        The decoded values as `SIMD[.float32, width]`, bit-identical to
        indexing `fp6_reference_table[fmt]()`.
    """
    comptime M = fmt.mantissa_width()
    comptime bias = fmt.exponent_bias()
    comptime mantissa_mask = (1 << M) - 1
    comptime exponent_mask = (1 << fmt.exponent_width()) - 1

    comptime subnormal_step = bitcast[.float32](
        UInt32((127 + 1 - bias - M) << 23)
    )

    var n = code.cast[.uint32]()
    comptime Bits = SIMD[.uint32, width]
    var e = (n >> Bits(M)) & Bits(exponent_mask)
    var m = n & Bits(mantissa_mask)
    var sign = (n & Bits(0x20)) << Bits(26)  # code bit 5 -> float32 bit 31

    var normal_bits = ((e + Bits(127 - bias)) << Bits(23)) | (m << Bits(23 - M))
    var subnormal_bits = bitcast[.uint32](m.cast[.float32]() * subnormal_step)

    var is_subnormal = e.eq(type_of(e)(0))
    var magnitude = is_subnormal.select(subnormal_bits, normal_bits)
    return bitcast[.float32](sign | magnitude)


@always_inline
def decode_fp6_to_bf16[
    width: SIMDLength, //, fmt: FP6Format
](code: SIMD[.uint8, width]) -> SIMD[.bfloat16, width]:
    """Decodes FP6 codes to bfloat16.

    Every FP6 value carries at most 3 mantissa bits and lives well inside
    bfloat16's exponent range, so the float32 decode followed by a bfloat16
    cast is exact -- the cast never rounds.

    Parameters:
        width: SIMD width (lane count) of the code vector.
        fmt: The FP6 encoding of the input codes.

    Args:
        code: One FP6 code per lane in the low 6 bits (`0..63`).

    Returns:
        The decoded values as `SIMD[.bfloat16, width]`.
    """
    return decode_fp6_to_f32[fmt](code).cast[.bfloat16]()


@always_inline
def encode_f32_to_fp6[
    width: SIMDLength, //, fmt: FP6Format
](x: SIMD[.float32, width]) -> SIMD[.uint8, width]:
    """Encodes float32 values to FP6 codes with round-to-nearest-even.

    Expects `x` to already be divided by its block's E8M0 scale. Magnitudes
    above the format maximum saturate; FP6 has no Inf or NaN encoding, so a
    non-finite input has no faithful representation and the caller must screen
    for it (an all-ones E8M0 block scale is the MX spec's NaN channel).

    Two regimes, selected branch-free:

    - Subnormal (`|x| < 2^(1 - bias)`): the representable values form a uniform
      grid of step `S = 2^(1 - bias - M)`, so adding a magic `2^23 * S` forces
      the hardware to round `|x|` onto that grid with the FPU's own
      round-to-nearest-even, and the difference from the magic's bit pattern is
      the mantissa code. `|x|` is bounded well below the magic, so the add
      never leaves the magic's binade.
    - Normal: round the float32 significand at bit `23 - M` with the standard
      round-to-nearest-even integer add, then read the exponent and mantissa
      fields off the rounded pattern. Rounding that carries into the exponent
      is handled for free because the carry propagates through the full bit
      pattern before extraction.

    Parameters:
        width: SIMD width (lane count) of the input vector.
        fmt: The FP6 encoding to produce.

    Args:
        x: Scale-normalized values, one per lane. Must be finite.

    Returns:
        One FP6 code per lane in the low 6 bits (`0..63`).
    """
    comptime M = fmt.mantissa_width()
    comptime bias = fmt.exponent_bias()
    comptime mantissa_mask = (1 << M) - 1
    comptime max_code = UInt32(
        (((1 << fmt.exponent_width()) - 1) << M) | mantissa_mask
    )
    comptime min_normal = bitcast[.float32](UInt32((127 + 1 - bias) << 23))
    comptime magic = bitcast[.float32](UInt32((127 + 23 + 1 - bias - M) << 23))
    comptime magic_bits = bitcast[.uint32](magic)
    comptime round_shift = 23 - M

    comptime Bits = SIMD[.uint32, width]
    comptime Floats = SIMD[.float32, width]

    var bits = bitcast[.uint32](x)
    var sign = (bits >> Bits(26)) & Bits(0x20)  # float32 bit 31 -> code bit 5
    var magnitude = bitcast[.float32](bits & Bits(0x7FFF_FFFF))

    magnitude = min(magnitude, Floats(fmt.max_value()))

    var subnormal_code = bitcast[.uint32](magnitude + Floats(magic)) - Bits(
        magic_bits
    )

    var magnitude_bits = bitcast[.uint32](magnitude)
    var lsb = (magnitude_bits >> Bits(round_shift)) & Bits(1)
    var rounded = magnitude_bits + (Bits((1 << (round_shift - 1)) - 1) + lsb)
    var normal_code = (
        ((rounded >> Bits(23)) - Bits(127 - bias)) << Bits(M)
    ) | ((rounded >> Bits(round_shift)) & Bits(mantissa_mask))

    var is_subnormal = magnitude.lt(Floats(min_normal))
    var code = is_subnormal.select(subnormal_code, normal_code)
    return (sign | min(code, Bits(max_code))).cast[.uint8]()


@always_inline
def compute_mxfp6_even_scale[
    fmt: FP6Format
](max_val: Float32) -> Float8_e8m0fnu:
    """Computes the OCP MXFP6 E8M0 scale using even-mode rounding.

    The FP6 generalization of `compute_mxfp4_even_scale`: it rounds the block
    maximum at the target's mantissa boundary before deriving the exponent, so
    a block whose maximum is just under a power-of-two boundary does not waste
    a whole binade of range on the other 31 elements.

    Parameters:
        fmt: The FP6 encoding the block will be quantized to.

    Args:
        max_val: The largest magnitude in the 32-element block.

    Returns:
        The E8M0 scale for the block.
    """
    comptime FP32_MANTISSA_WIDTH = FPUtils[.float32].mantissa_width()
    comptime round_to_fp6_mantissa = 1 << (
        FP32_MANTISSA_WIDTH - fmt.mantissa_width() - 1
    )

    var max_bits = FPUtils[.float32].bitcast_to_uint(max_val)
    var rounded_max_bits = max_bits + type_of(max_bits)(round_to_fp6_mantissa)
    var rounded_max = bitcast[.float32](rounded_max_bits)
    var scale_exp = (
        FPUtils[.float32].get_exponent_biased(rounded_max) - fmt.max_exponent()
    )
    scale_exp = max(0, min(scale_exp, 254))
    return bitcast[.float8_e8m0fnu](UInt8(scale_exp))


@always_inline
def pack_fp6_x4(code: SIMD[.uint8, 4]) -> UInt32:
    """Packs four FP6 codes into the low 24 bits of a word, element 0 lowest.

    Element `i` occupies bits `[6i + 5 : 6i]`. Storing the low three bytes of
    the result little-endian lays the group out as a contiguous 6-bit stream,
    which is why the group -- not the byte -- is the smallest addressable unit
    of packed FP6. The upper 8 bits of the returned word are always zero.

    A `SIMD[.uint8, 3]` would model the three bytes more literally, but
    SIMD lengths must be powers of two, so the group travels in a word instead.

    !!! warning "Bit order is unverified against the CDNA4 MFMA"
        This is the natural LSB-first reading of a packed FP6 stream, and it
        round-trips with `unpack_fp6_x4`, but nothing here proves it matches
        the operand layout `V_MFMA_SCALE_*_F8F6F4` expects inside its 24-byte
        fragment. Only an MFMA run on MI355 settles that; do not build a kernel
        fragment loader on this convention until such a test passes.

    Args:
        code: Four FP6 codes, each in the low 6 bits.

    Returns:
        The packed group in bits 23:0.
    """
    var c = code.cast[.uint32]() & 0x3F
    return c[0] | (c[1] << 6) | (c[2] << 12) | (c[3] << 18)


@always_inline
def unpack_fp6_x32(fragment: SIMD[.uint8, 32]) -> SIMD[.uint8, 32]:
    """Unpacks the 24 payload bytes of a 32-byte fragment into 32 FP6 codes.

    Thirty-two elements is one MX block, so this is the natural unit for both
    the dequant kernel (one E8M0 scale covers exactly this many elements) and
    an MFMA lane (which holds exactly this many). The 24 payload bytes travel
    in a 32-byte fragment because SIMD lengths must be powers of two -- the
    same reason `cdna4_block_scaled_mfma` declares a 32-byte operand. Bytes
    24..31 are ignored.

    Args:
        fragment: 24 packed payload bytes in lanes 0..23; upper lanes ignored.

    Returns:
        Thirty-two FP6 codes, each in the low 6 bits.
    """
    var codes = SIMD[.uint8, 32](0)
    comptime for group in range(8):
        comptime b = 3 * group
        var word = (
            UInt32(fragment[b])
            | (UInt32(fragment[b + 1]) << 8)
            | (UInt32(fragment[b + 2]) << 16)
        )
        codes = codes.insert[offset=4 * group](unpack_fp6_x4(word))
    return codes


@always_inline
def unpack_fp6_x4(packed: UInt32) -> SIMD[.uint8, 4]:
    """Unpacks a 24-bit group into four FP6 codes, the inverse of `pack_fp6_x4`.

    Args:
        packed: Four packed FP6 codes in bits 23:0. Upper bits are ignored.

    Returns:
        Four FP6 codes, each in the low 6 bits.
    """
    var lanes = SIMD[.uint32, 4](packed)
    var shifts = SIMD[.uint32, 4](0, 6, 12, 18)
    return ((lanes >> shifts) & SIMD[.uint32, 4](0x3F)).cast[.uint8]()
