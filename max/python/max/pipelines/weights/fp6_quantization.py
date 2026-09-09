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
"""Host-side OCP MXFP6 weight quantization.

The numpy counterpart of ``max/kernels/src/linalg/fp6_utils.mojo``: same E8M0
even-mode scale derivation, same round-to-nearest-even element encoding, same
four-codes-per-three-bytes packing. Used to requantize a bf16 or MXFP8
checkpoint to MXFP6 offline (see ``_quantize_checkpoint.py``) and to
RTN-quantize at load time.

Both modules must agree bit-for-bit, since a checkpoint written here is decoded
by the kernel there. ``max/tests/.../test_fp6_quantization.py`` pins that by
checking this encoder against the 64-entry reference tables that
``fp6_utils.mojo`` transcribes by hand from the OCP MX specification.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

MX_BLOCK_SIZE = 32
"""Elements sharing one E8M0 scale, fixed by the OCP MX specification."""

_CODES_PER_GROUP = 4
"""FP6 codes per packed group. Four 6-bit codes tile three bytes exactly."""

_BYTES_PER_GROUP = 3


class FP6Format(Enum):
    """Selects between the two OCP MX FP6 element encodings.

    Mirrors ``FP6Format`` in ``fp6_utils.mojo``; the numeric parameters below
    are the entire difference between the two encodings.
    """

    E2M3 = "e2m3"
    E3M2 = "e3m2"

    @property
    def mantissa_width(self) -> int:
        """The mantissa field width in bits."""
        return 3 if self is FP6Format.E2M3 else 2

    @property
    def exponent_width(self) -> int:
        """The exponent field width in bits."""
        return 2 if self is FP6Format.E2M3 else 3

    @property
    def exponent_bias(self) -> int:
        """The exponent bias."""
        return 1 if self is FP6Format.E2M3 else 3

    @property
    def max_exponent(self) -> int:
        """The unbiased exponent of the largest finite value.

        E2M3 tops out at ``7.5 == 1.875 * 2**2`` and E3M2 at
        ``28.0 == 1.75 * 2**4``.
        """
        return 2 if self is FP6Format.E2M3 else 4

    @property
    def max_value(self) -> float:
        """The largest finite magnitude the encoding represents."""
        return 7.5 if self is FP6Format.E2M3 else 28.0


def encode_f32_to_fp6(
    x: NDArray[np.float32], fmt: FP6Format
) -> NDArray[np.uint8]:
    """Encodes float32 values to FP6 codes with round-to-nearest-even.

    Expects ``x`` to already be divided by its block's E8M0 scale. Magnitudes
    above the format maximum saturate. FP6 has no Inf or NaN encoding, so a
    non-finite input has no faithful representation and the caller must screen
    for it.

    Two regimes, selected branch-free. Below ``2**(1 - bias)`` the
    representable values form a uniform grid, so adding a magic constant makes
    the FPU round onto that grid with its own round-to-nearest-even and the
    difference from the magic's bit pattern is the code. Above it, round the
    float32 significand at bit ``23 - M`` and read the exponent and mantissa
    fields off the rounded pattern; a rounding carry into the exponent is
    handled for free because it propagates before extraction.

    Args:
        x: Scale-normalized values. Must be finite.
        fmt: The FP6 encoding to produce.

    Returns:
        One FP6 code per element, in the low 6 bits.
    """
    m = fmt.mantissa_width
    bias = fmt.exponent_bias
    mantissa_mask = np.uint32((1 << m) - 1)
    max_code = np.uint32(
        (((1 << fmt.exponent_width) - 1) << m) | int(mantissa_mask)
    )
    min_normal = _f32_from_bits(np.uint32((127 + 1 - bias) << 23))
    magic_bits = np.uint32((127 + 23 + 1 - bias - m) << 23)
    magic = _f32_from_bits(magic_bits)
    round_shift = np.uint32(23 - m)

    values = np.ascontiguousarray(x, dtype=np.float32)
    bits = values.view(np.uint32)
    sign = (bits >> np.uint32(26)) & np.uint32(0x20)  # f32 bit 31 -> code bit 5
    magnitude = (bits & np.uint32(0x7FFF_FFFF)).view(np.float32)

    # Saturating first keeps the round-to-nearest add below from carrying out
    # of the exponent field; values between the max and the rounding threshold
    # would saturate anyway, so this does not perturb any rounding decision.
    magnitude = np.minimum(magnitude, np.float32(fmt.max_value))

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


def compute_mx_even_scale(
    block_max: NDArray[np.float32], fmt: FP6Format
) -> NDArray[np.uint8]:
    """Computes OCP MXFP6 E8M0 scale bytes using even-mode rounding.

    Rounds the block maximum at the target's mantissa boundary before deriving
    the exponent, so a block whose maximum sits just under a power-of-two
    boundary does not waste a whole binade of range on the other 31 elements.
    This is ``scale_calculation_mode: "even"`` in a Quark config, and the port
    of ``compute_mxfp6_even_scale``.

    Args:
        block_max: The largest magnitude in each 32-element block.
        fmt: The FP6 encoding the blocks will be quantized to.

    Returns:
        One E8M0 scale byte per block, representing ``2**(byte - 127)``.
    """
    round_to_fp6_mantissa = np.uint32(1 << (23 - fmt.mantissa_width - 1))

    max_bits = np.ascontiguousarray(block_max, dtype=np.float32).view(np.uint32)
    with np.errstate(over="ignore"):
        rounded_bits = max_bits + round_to_fp6_mantissa
    biased_exponent = (rounded_bits >> np.uint32(23)) & np.uint32(0xFF)

    scale_exp = biased_exponent.astype(np.int32) - fmt.max_exponent
    return np.clip(scale_exp, 0, 254).astype(np.uint8)


def pack_fp6(codes: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Packs FP6 codes along the last axis, four codes to three bytes.

    Element ``i`` of a group occupies bits ``[6i + 5 : 6i]`` of a 24-bit word
    stored little-endian, which lays a group out as a contiguous 6-bit stream.
    The group, not the byte, is therefore the smallest addressable unit of
    packed FP6. Port of ``pack_fp6_x4``.

    Args:
        codes: FP6 codes in the low 6 bits; the last axis must be a multiple
            of 4.

    Returns:
        Packed bytes, last axis scaled by 3/4.
    """
    if codes.shape[-1] % _CODES_PER_GROUP:
        raise ValueError(
            f"FP6 packing needs a multiple of {_CODES_PER_GROUP} codes on the "
            f"last axis, got {codes.shape[-1]}"
        )

    groups = codes.reshape(*codes.shape[:-1], -1, _CODES_PER_GROUP).astype(
        np.uint32
    )
    groups &= np.uint32(0x3F)
    word = (
        groups[..., 0]
        | (groups[..., 1] << np.uint32(6))
        | (groups[..., 2] << np.uint32(12))
        | (groups[..., 3] << np.uint32(18))
    )

    packed = np.stack(
        [
            (word & np.uint32(0xFF)).astype(np.uint8),
            ((word >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8),
            ((word >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8),
        ],
        axis=-1,
    )
    return packed.reshape(*codes.shape[:-1], -1)


def unpack_fp6(packed: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Unpacks FP6 codes along the last axis, three bytes to four codes.

    Inverse of :func:`pack_fp6`.

    Args:
        packed: Packed bytes; the last axis must be a multiple of 3.

    Returns:
        FP6 codes in the low 6 bits, last axis scaled by 4/3.
    """
    if packed.shape[-1] % _BYTES_PER_GROUP:
        raise ValueError(
            f"FP6 unpacking needs a multiple of {_BYTES_PER_GROUP} bytes on "
            f"the last axis, got {packed.shape[-1]}"
        )

    groups = packed.reshape(*packed.shape[:-1], -1, _BYTES_PER_GROUP).astype(
        np.uint32
    )
    word = (
        groups[..., 0]
        | (groups[..., 1] << np.uint32(8))
        | (groups[..., 2] << np.uint32(16))
    )

    codes = np.stack(
        [(word >> np.uint32(6 * i)) & np.uint32(0x3F) for i in range(4)],
        axis=-1,
    ).astype(np.uint8)
    return codes.reshape(*packed.shape[:-1], -1)


def fp6_decode_table(fmt: FP6Format) -> NDArray[np.float32]:
    """Builds the 64-entry FP6 code-to-float32 table.

    Args:
        fmt: The FP6 encoding to tabulate.

    Returns:
        The value of every 6-bit code, indexed by the code.
    """
    m = fmt.mantissa_width
    bias = fmt.exponent_bias
    codes = np.arange(64, dtype=np.uint32)

    exponent = (codes >> np.uint32(m)) & np.uint32(
        (1 << fmt.exponent_width) - 1
    )
    mantissa = (codes & np.uint32((1 << m) - 1)).astype(np.float32)
    sign = np.where(codes & np.uint32(0x20), np.float32(-1.0), np.float32(1.0))

    subnormal = mantissa * np.float32(2.0 ** (1 - bias - m))
    normal = (np.float32(1.0) + mantissa * np.float32(2.0**-m)) * np.exp2(
        exponent.astype(np.float32) - np.float32(bias)
    )
    return (sign * np.where(exponent == 0, subnormal, normal)).astype(
        np.float32
    )


def quantize_mxfp6(
    weight: NDArray[np.floating], fmt: FP6Format
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Quantizes a weight tensor to MXFP6 along its last axis.

    Blocks of 32 consecutive elements on the last axis share one E8M0 scale,
    matching the ``[1, 32]`` weight block size the AMD block-scaled kernels
    read. Values are normalized by an exact power of two, so the division
    itself introduces no error.

    Args:
        weight: The weight tensor; its last axis must be a multiple of 32.
        fmt: The FP6 encoding to produce.

    Returns:
        ``(packed, scales)`` where ``packed`` is uint8 ``[..., K * 3 // 4]``
        and ``scales`` is uint8 E8M0 ``[..., K // 32]``.

    Raises:
        ValueError: If the last axis is not a multiple of 32, or the input is
            not finite.
    """
    if weight.shape[-1] % MX_BLOCK_SIZE:
        raise ValueError(
            f"MXFP6 needs a K that is a multiple of {MX_BLOCK_SIZE}, got "
            f"{weight.shape[-1]}"
        )

    values = weight.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(
            "MXFP6 has no Inf or NaN encoding; the source weight is not finite"
        )

    blocks = values.reshape(*values.shape[:-1], -1, MX_BLOCK_SIZE)
    scales = compute_mx_even_scale(
        np.abs(blocks).max(axis=-1).astype(np.float32), fmt
    )

    # Multiply by 2**(127 - scale) instead of dividing by the E8M0 value: a
    # scale byte of 0 denotes 2**-127, which is subnormal in float32 and would
    # overflow the quotient.
    normalized = np.ldexp(
        blocks, (np.int32(127) - scales.astype(np.int32))[..., None]
    )
    codes = encode_f32_to_fp6(normalized.astype(np.float32), fmt)

    packed = pack_fp6(codes.reshape(*values.shape[:-1], -1))
    return packed, scales


def dequantize_mxfp6(
    packed: NDArray[np.uint8], scales: NDArray[np.uint8], fmt: FP6Format
) -> NDArray[np.float32]:
    """Reconstructs float32 values from packed MXFP6 weights and E8M0 scales.

    The exact inverse of :func:`quantize_mxfp6` up to the quantization itself.
    Used to measure quantization error and as the reference for tests.

    Args:
        packed: Packed FP6 bytes ``[..., K * 3 // 4]``.
        scales: E8M0 scale bytes ``[..., K // 32]``.
        fmt: The FP6 encoding the bytes hold.

    Returns:
        The dequantized values, ``[..., K]``.
    """
    codes = unpack_fp6(packed)
    values = fp6_decode_table(fmt)[codes]

    blocks = values.reshape(*values.shape[:-1], -1, MX_BLOCK_SIZE)
    scaled = np.ldexp(
        blocks, (scales.astype(np.int32) - np.int32(127))[..., None]
    )
    return scaled.reshape(*values.shape).astype(np.float32)


def _f32_from_bits(bits: np.uint32) -> np.float32:
    """Reinterprets a uint32 bit pattern as float32."""
    return np.float32(np.array(bits, dtype=np.uint32).view(np.float32)[()])
