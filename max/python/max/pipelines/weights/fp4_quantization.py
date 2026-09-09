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
"""Host-side NVFP4 weight quantization.

The numpy counterpart of ``max/kernels/src/linalg/fp4_utils.mojo``: same E2M1
round-to-nearest-even, same E4M3 block scales, same two-codes-per-byte packing.
Used to requantize a bf16 checkpoint to NVFP4 offline (see
``quantize_checkpoint.py``).

NVFP4 and a future MXFP4 share the E2M1 element encoding. They differ in block
size and scale scheme: NVFP4 uses 16-wide E4M3 block scales plus a global
float32 tensor scale; MXFP4 uses 32-wide E8M0 scales.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np
from numpy.typing import NDArray

BLOCK_SIZE = 16
MAX_E2M1 = 6.0
MAX_E4M3 = 448.0
MAX_FP32 = np.finfo(np.float32).max


class FP4Format(Enum):
    """Selects the FP4 scale scheme. Element codes are always E2M1.

    NVFP4 and a future MXFP4 share the same 4-bit element encoding. They
    differ in block size and how scales are stored: NVFP4 uses 16-wide E4M3
    block scales plus a global tensor scale; MXFP4 uses 32-wide E8M0 scales.
    """

    NVFP4 = "nvfp4"

    @property
    def mantissa_width(self) -> int:
        """The E2M1 mantissa field width in bits."""
        return 1

    @property
    def exponent_width(self) -> int:
        """The E2M1 exponent field width in bits."""
        return 2

    @property
    def exponent_bias(self) -> int:
        """The E2M1 exponent bias."""
        return 1

    @property
    def max_exponent(self) -> int:
        """The unbiased exponent of the largest finite E2M1 value."""
        return 2

    @property
    def max_value(self) -> float:
        """The largest finite E2M1 magnitude (``6.0 == 1.5 * 2**2``)."""
        return MAX_E2M1

    @property
    def block_size(self) -> int:
        """Elements sharing one block scale."""
        return 16 if self is FP4Format.NVFP4 else 32

    @property
    def has_global_scale(self) -> bool:
        """Whether a tensor-level scale sits beside the per-block scales."""
        return self is FP4Format.NVFP4

    @property
    def scale_max(self) -> float:
        """Largest finite magnitude of the per-block scale encoding."""
        return MAX_E4M3 if self.has_global_scale else 1.0


def _f32_from_bits(bits: np.uint32) -> np.float32:
    """Reinterprets a uint32 bit pattern as float32."""
    return np.float32(np.array(bits, dtype=np.uint32).view(np.float32)[()])


def encode_f32_to_e4m3(x: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Encodes float32 values to E4M3 with round-to-nearest-even.

    Saturates at ``±448``. The all-ones code is NaN in E4M3, so the largest
    finite code is used instead. Same two-regime construction as
    ``encode_f32_to_fp6``.
    """
    m = 3
    bias = 7
    mantissa_mask = np.uint32((1 << m) - 1)
    # exp=15, mant=7 is NaN; saturate to the largest finite code.
    max_code = np.uint32((((1 << 4) - 1) << m) | 6)
    min_normal = _f32_from_bits(np.uint32((127 + 1 - bias) << 23))
    magic_bits = np.uint32((127 + 23 + 1 - bias - m) << 23)
    magic = _f32_from_bits(magic_bits)
    round_shift = np.uint32(23 - m)

    values = np.ascontiguousarray(x, dtype=np.float32)
    bits = values.view(np.uint32)
    sign = (bits >> np.uint32(24)) & np.uint32(0x80)
    magnitude = (bits & np.uint32(0x7FFF_FFFF)).view(np.float32)
    magnitude = np.minimum(magnitude, np.float32(MAX_E4M3))

    with np.errstate(over="ignore"):
        subnormal_code = (magnitude + magic).view(np.uint32) - magic_bits
        magnitude_bits = magnitude.view(np.uint32)
        lsb = (magnitude_bits >> round_shift) & np.uint32(1)
        rounded = magnitude_bits + (
            np.uint32((1 << (int(round_shift) - 1)) - 1) + lsb
        )
        normal_code = (
            ((rounded >> np.uint32(23)) - np.uint32(127 - bias)) << np.uint32(m)
        ) | ((rounded >> round_shift) & mantissa_mask)

    code = np.where(magnitude < min_normal, subnormal_code, normal_code)
    return (sign | np.minimum(code, max_code)).astype(np.uint8)


def unpack_e4m3_uint8_to_fp32(
    uint8_arr: NDArray[np.uint8],
) -> NDArray[np.float32]:
    """Decodes E4M3 bytes to float32.

    Inverse of :func:`encode_f32_to_e4m3` for finite codes. The all-ones NaN
    code is not produced by the encoder. ``exp == 0`` is the subnormal class
    ``mant * 2**(1 - bias - m)`` (``mant * 2**-9``), not a normal with
    exponent ``-7``.

    Args:
        uint8_arr: Packed E4M3 codes.

    Returns:
        The decoded values, same shape as ``uint8_arr``.
    """
    packed = uint8_arr.astype(np.uint32)
    sign = (packed >> 7) & 0x1
    exp_e4m3 = (packed >> 3) & 0xF
    mant_e4m3 = packed & 0x7

    # E == 0 is subnormal: 2^(1-7) * (mant/8) = mant * 2^-9. The normal-path
    # rebasing (exp+120, mant<<20) would decode mant=1 as ~0.0088.
    is_subnormal = exp_e4m3 == 0
    normal_bits = ((exp_e4m3 + 120) << 23) | (mant_e4m3 << 20)
    subnormal_bits = (mant_e4m3.astype(np.float32) * np.float32(2.0**-9)).view(
        np.uint32
    )
    mag_bits = np.where(is_subnormal, subnormal_bits, normal_bits)
    return ((sign << 31) | mag_bits).view(np.float32)


def round_to_e2m1(x: NDArray[np.floating]) -> NDArray[np.float32]:
    """Rounds each element to the nearest E2M1 value, ties to even.

    Mirrors ``cast_fp_to_fp4e2m1`` in ``fp4_utils.mojo``. The representable
    magnitudes are ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}``; values outside ``[-6, 6]``
    saturate.
    """
    values = np.ascontiguousarray(x, dtype=np.float32)
    sign = np.where(values < 0, np.float32(-1.0), np.float32(1.0))
    abs_x = np.abs(values)
    rounded = np.select(
        [
            abs_x <= 0.25,
            abs_x < 0.75,
            abs_x <= 1.25,
            abs_x < 1.75,
            abs_x <= 2.5,
            abs_x < 3.5,
            abs_x <= 5.0,
        ],
        [
            np.float32(0.0),
            np.float32(0.5),
            np.float32(1.0),
            np.float32(1.5),
            np.float32(2.0),
            np.float32(3.0),
            np.float32(4.0),
        ],
        default=np.float32(6.0),
    )
    return (sign * rounded).astype(np.float32)


def pack_pairs_e2m1_uint8(
    fp32_arr: NDArray[np.floating],
) -> NDArray[np.uint8]:
    """Packs E2M1 values two-to-a-byte along a flat last axis.

    Expects values already rounded onto the E2M1 grid. Even indices occupy the
    low nibble, odd indices the high nibble — the same lo-nibble-first layout
    ``cast_fp32_to_fp4e2m1`` writes.

    Args:
        fp32_arr: Rounded E2M1 values. An odd last element is padded with 0.

    Returns:
        Packed bytes, half as long as the (possibly padded) input.
    """
    # Ensure the input array has an even number of elements for pairing
    if fp32_arr.size % 2 != 0:
        fp32_arr = np.append(fp32_arr, 0.0)

    # 1. View raw underlying 32-bit bits
    fp32_bits = fp32_arr.astype(np.float32).view(np.uint32)

    # 2. Extract components from FP32
    sign = (fp32_bits >> 31) & 0x1
    exp_fp32 = (fp32_bits >> 23) & 0xFF
    mant_fp32 = fp32_bits & 0x7FFFFF

    # 3. E2M1: normals use exp = fp32_exp - 126; |x| < 1 is the subnormal
    # class (0.0 or 0.5). Bit extraction without this branch encodes 0.5 as 0.
    is_subnormal = exp_fp32 < 127
    exp_e2m1 = np.where(
        is_subnormal,
        0,
        np.clip(exp_fp32.astype(np.int32) - 126, 0, 3),
    ).astype(np.uint32)
    mant_e2m1 = np.where(
        is_subnormal,
        (exp_fp32 != 0) | (mant_fp32 != 0),
        (mant_fp32 >> 22) & 0x1,
    ).astype(np.uint32)

    # 4. Build individual 4-bit items (0000SEEM)
    four_bit_elements = (sign << 3) | (exp_e2m1 << 1) | mant_e2m1

    # 6. Pair them up: even indices on the bottom, odd indices on the top
    evens = four_bit_elements[0::2]
    odds = four_bit_elements[1::2]

    # Pack into one byte: [ odds (4-bits) | evens (4-bits) ]
    packed_uint8 = (odds << 4) | evens

    return packed_uint8.astype(np.uint8)


def unpack_pairs_e2m1_to_fp32(
    packed_uint8: NDArray[np.uint8],
) -> NDArray[np.float32]:
    """Unpacks two-to-a-byte E2M1 codes to float32.

    Inverse of :func:`pack_pairs_e2m1_uint8`. Matches ``decode_e2m1_to_f32``
    in ``fp4_utils.mojo``, including the subnormal ``0.5`` class.

    Args:
        packed_uint8: Packed E2M1 bytes (low nibble = even index).

    Returns:
        Decoded values, twice as long as ``packed_uint8``.
    """
    # Upcast to uint32 to safely execute 32-bit shifting routines
    packed = packed_uint8.astype(np.uint32)

    # 1. Separate the interleaved streams back out
    evens = packed & 0x0F  # Lower 4 bits
    odds = (packed >> 4) & 0x0F  # Upper 4 bits

    # 2. Reconstruct a flat 4-bit array in the correct chronological order
    four_bit_elements = np.empty(packed.size * 2, dtype=np.uint32)
    four_bit_elements[0::2] = evens
    four_bit_elements[1::2] = odds

    # 3. Extract standard E2M1 structures from the 4-bit blocks
    sign = (four_bit_elements >> 3) & 0x1
    exp_e2m1 = (four_bit_elements >> 1) & 0x3
    mant_e2m1 = four_bit_elements & 0x1

    # 4. Re-bias. E == 0 is the subnormal class: 0.0 or 0.5, not a normal
    # with exponent 126 (which would decode 0.5 as 0.75).
    is_subnormal = exp_e2m1 == 0
    normal_bits = ((exp_e2m1 + 126) << 23) | (mant_e2m1 << 22)
    subnormal_bits = np.where(mant_e2m1, np.uint32(0x3F000000), np.uint32(0))
    mag_bits = np.where(is_subnormal, subnormal_bits, normal_bits)
    fp32_bits = (sign << 31) | mag_bits

    return fp32_bits.view(np.float32)


def e2m1_decode_table(fmt: FP4Format) -> NDArray[np.float32]:
    """Builds the 16-entry E2M1 code-to-float32 table.

    Args:
        fmt: The FP4 encoding to tabulate.

    Returns:
        The value of every 4-bit code, indexed by the code.
    """
    del fmt  # NVFP4 currently has a single element encoding.
    codes = np.arange(16, dtype=np.uint32)
    sign = (codes >> np.uint32(3)) & np.uint32(1)
    exp_e2m1 = (codes >> np.uint32(1)) & np.uint32(3)
    mant_e2m1 = codes & np.uint32(1)
    is_subnormal = exp_e2m1 == 0
    normal_bits = ((exp_e2m1 + np.uint32(126)) << np.uint32(23)) | (
        mant_e2m1 << np.uint32(22)
    )
    subnormal_bits = np.where(mant_e2m1, np.uint32(0x3F000000), np.uint32(0))
    mag_bits = np.where(is_subnormal, subnormal_bits, normal_bits)
    return ((sign << np.uint32(31)) | mag_bits).view(np.float32)


def quantize_nvfp4(
    weight: NDArray[np.floating], fmt: FP4Format
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], float]:
    """Quantizes a weight tensor to NVFP4 along its last axis.

    Returns packed E2M1 weights, E4M3 per-block decode scales, and a global
    float32 decode scale.
    """
    if not fmt.has_global_scale:
        raise ValueError(f"{fmt.value} is not implemented by quantize_nvfp4")
    if weight.shape[-1] % fmt.block_size:
        raise ValueError(
            f"NVFP4 needs a K that is a multiple of {fmt.block_size}, got "
            f"{weight.shape[-1]}"
        )

    dims = weight.shape
    amax = np.abs(weight).max()
    encode_scale = (fmt.max_value * fmt.scale_max) / amax
    encode_scale = min(encode_scale, MAX_FP32)
    if encode_scale == 0.0:
        encode_scale = 1.0

    decode_scale = 1.0 / encode_scale

    block_weight = weight.reshape(
        list(dims[:-1]) + [dims[-1] // fmt.block_size, fmt.block_size]
    )
    block_amax = np.abs(block_weight).max(axis=-1, keepdims=True)
    block_decode_scales = block_amax / fmt.max_value

    block_decode_scales_fp8 = encode_f32_to_e4m3(
        block_decode_scales * encode_scale
    )

    block_encode_scales = 1.0 / (
        unpack_e4m3_uint8_to_fp32(block_decode_scales_fp8) * decode_scale
    )
    block_encode_scales = np.minimum(block_encode_scales, MAX_FP32)

    block_weight_fp4 = round_to_e2m1(
        block_weight.astype(np.float32) * block_encode_scales
    )

    flat = block_weight_fp4.reshape(*dims).ravel()
    packed = pack_pairs_e2m1_uint8(flat).reshape(*dims[:-1], dims[-1] // 2)
    return packed, block_decode_scales_fp8, decode_scale


def dequantize_nvfp4(
    packed: NDArray[np.uint8],
    block_decode_scales_fp8: NDArray[np.uint8],
    decode_scale: float,
    fmt: FP4Format,
) -> NDArray[np.float32]:
    """Reconstructs float32 values from packed NVFP4 weights and scales."""
    if not fmt.has_global_scale:
        raise ValueError(f"{fmt.value} is not implemented by dequantize_nvfp4")

    dims = (*packed.shape[:-1], packed.shape[-1] * 2)
    weight_fp4 = unpack_pairs_e2m1_to_fp32(packed.ravel()).reshape(*dims)
    block_weight_fp4 = weight_fp4.reshape(
        list(dims[:-1]) + [dims[-1] // fmt.block_size, fmt.block_size]
    )
    block_decode_scales = unpack_e4m3_uint8_to_fp32(block_decode_scales_fp8)
    return (block_weight_fp4 * block_decode_scales * decode_scale).reshape(
        *dims
    )


def nvfp4_quantization_config(
    fmt: FP4Format, ignored: Sequence[str]
) -> dict[str, object]:
    """Builds the ``quantization_config`` block for an NVFP4 checkpoint.

    ``quant_method`` is ``modelopt`` so serving takes the same parse path as
    an upstream ModelOpt dump. ``quant_algo`` is still ``NVFP4``.
    """
    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "activation_scheme": "dynamic",
        "weight_block_size": [1, fmt.block_size],
        "ignored_layers": list(ignored),
    }
