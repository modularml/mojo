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
"""Tests the host-side MXFP6 encoder against the OCP MX specification.

The kernel decodes what this module encodes, so the two must agree bit-for-bit.
The reference tables below are transcribed from the OCP MX specification --
the same hand-written values ``max/kernels/src/linalg/fp6_utils.mojo`` pins its
arithmetic decoder against. Deriving them from the code under test would make
these assertions vacuous, which is the whole reason they are spelled out.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.pipelines.weights.fp6_quantization import (
    MX_BLOCK_SIZE,
    FP6Format,
    compute_mx_even_scale,
    dequantize_mxfp6,
    encode_f32_to_fp6,
    fp6_decode_table,
    pack_fp6,
    quantize_mxfp6,
    unpack_fp6,
)

# Codes 0..31 of each encoding; codes 32..63 are their negations.
E2M3_POSITIVE = [
    0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875,
    1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
    2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
    4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
]  # fmt: skip

E3M2_POSITIVE = [
    0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,
    0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75,
    2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0,
    8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0,
]  # fmt: skip

_REFERENCE = {
    FP6Format.E2M3: E2M3_POSITIVE,
    FP6Format.E3M2: E3M2_POSITIVE,
}

ALL_FORMATS = pytest.mark.parametrize("fmt", list(FP6Format))


@ALL_FORMATS
def test_decode_table_matches_ocp_spec(fmt: FP6Format) -> None:
    table = fp6_decode_table(fmt)
    assert table.shape == (64,)
    np.testing.assert_array_equal(
        table[:32], np.array(_REFERENCE[fmt], dtype=np.float32)
    )
    # Sign is bit 5, so the negative half mirrors the positive half exactly.
    np.testing.assert_array_equal(table[32:], -table[:32])
    assert table.max() == np.float32(fmt.max_value)


@ALL_FORMATS
def test_representable_values_encode_exactly(fmt: FP6Format) -> None:
    """Every value the format represents must survive a round trip."""
    table = fp6_decode_table(fmt)
    codes = encode_f32_to_fp6(table, fmt)
    np.testing.assert_array_equal(table[codes], table)


@ALL_FORMATS
def test_midpoints_round_to_even(fmt: FP6Format) -> None:
    """``round_method: half_even``, the mode the MX checkpoints declare.

    A tie exactly between two representable values must land on the one with
    an even code, not away from zero.
    """
    positive = np.unique(np.array(_REFERENCE[fmt], dtype=np.float32))
    midpoints = ((positive[:-1] + positive[1:]) / 2).astype(np.float32)
    codes = encode_f32_to_fp6(midpoints, fmt)
    assert (codes % 2 == 0).all()


@ALL_FORMATS
def test_saturates_and_preserves_sign(fmt: FP6Format) -> None:
    table = fp6_decode_table(fmt)
    inputs = np.array([1e30, -1e30, fmt.max_value * 2, -0.0], dtype=np.float32)
    decoded = table[encode_f32_to_fp6(inputs, fmt)]
    np.testing.assert_array_equal(
        decoded,
        np.array(
            [fmt.max_value, -fmt.max_value, fmt.max_value, -0.0],
            dtype=np.float32,
        ),
    )
    # Negative zero must keep its sign bit rather than collapsing to +0.
    assert np.signbit(decoded[3])


@ALL_FORMATS
def test_pack_unpack_round_trip(fmt: FP6Format) -> None:
    codes = np.arange(64, dtype=np.uint8).reshape(2, 32)
    packed = pack_fp6(codes)
    assert packed.shape == (2, 24)
    np.testing.assert_array_equal(unpack_fp6(packed), codes)


def test_pack_bit_order() -> None:
    """Element ``i`` occupies bits ``[6i+5:6i]`` of a little-endian 24-bit word.

    Pinned explicitly because the packing convention -- not the byte -- is what
    the CDNA4 MFMA fragment layout depends on.
    """
    packed = pack_fp6(np.array([[0x3F, 0x00, 0x00, 0x00]], dtype=np.uint8))
    np.testing.assert_array_equal(
        packed, np.array([[0x3F, 0x00, 0x00]], np.uint8)
    )

    packed = pack_fp6(np.array([[0x00, 0x01, 0x00, 0x00]], dtype=np.uint8))
    np.testing.assert_array_equal(
        packed, np.array([[0x40, 0x00, 0x00]], np.uint8)
    )

    packed = pack_fp6(np.array([[0x00, 0x00, 0x00, 0x3F]], dtype=np.uint8))
    np.testing.assert_array_equal(
        packed, np.array([[0x00, 0x00, 0xFC]], np.uint8)
    )


def test_pack_rejects_ragged_last_axis() -> None:
    with pytest.raises(ValueError, match="multiple of 4"):
        pack_fp6(np.zeros((2, 6), dtype=np.uint8))


@ALL_FORMATS
def test_even_scale_uses_full_range(fmt: FP6Format) -> None:
    """The block maximum must map onto the format maximum, not below it."""
    for exponent in range(-20, 20):
        block_max = np.array(
            [np.ldexp(fmt.max_value, exponent)], dtype=np.float32
        )
        scale = compute_mx_even_scale(block_max, fmt)
        normalized = np.ldexp(block_max, 127 - int(scale[0]))
        assert normalized[0] == pytest.approx(fmt.max_value)


@ALL_FORMATS
def test_quantize_shapes(fmt: FP6Format) -> None:
    weight = np.zeros((8, 256), dtype=np.float32)
    packed, scales = quantize_mxfp6(weight, fmt)
    assert packed.shape == (8, 256 * 3 // 4)
    assert scales.shape == (8, 256 // MX_BLOCK_SIZE)
    assert packed.dtype == scales.dtype == np.uint8


@ALL_FORMATS
def test_quantize_is_exact_for_representable_values(fmt: FP6Format) -> None:
    """A tensor already on the format's grid must quantize losslessly.

    The scale is an exact power of two and every value is representable, so any
    error here would be an encoder bug rather than quantization noise.
    """
    rng = np.random.default_rng(0)
    table = fp6_decode_table(fmt)
    weight = table[rng.integers(0, 64, (4, 64))].astype(np.float32)
    packed, scales = quantize_mxfp6(weight, fmt)
    np.testing.assert_array_equal(dequantize_mxfp6(packed, scales, fmt), weight)


@ALL_FORMATS
def test_zero_block(fmt: FP6Format) -> None:
    packed, scales = quantize_mxfp6(np.zeros((1, 32), np.float32), fmt)
    assert (dequantize_mxfp6(packed, scales, fmt) == 0).all()


@ALL_FORMATS
def test_blocks_are_scaled_independently(fmt: FP6Format) -> None:
    """Neighbouring blocks with wildly different magnitudes must not interact.

    Sharing one scale across the row would flush the small block to zero, which
    is exactly what a per-32 block scale exists to prevent.
    """
    weight = np.concatenate(
        [
            np.full(32, 1e-4, dtype=np.float32),
            np.full(32, 1e4, dtype=np.float32),
        ]
    ).reshape(1, 64)
    packed, scales = quantize_mxfp6(weight, fmt)
    decoded = dequantize_mxfp6(packed, scales, fmt)
    assert scales[0, 0] != scales[0, 1]
    np.testing.assert_allclose(decoded, weight, rtol=0.2)


def test_e2m3_beats_e3m2_on_gaussian_weights() -> None:
    """E2M3's extra mantissa bit is why it is the default for weights.

    A per-32 block scale already absorbs dynamic range, so precision inside the
    block is what remains, and E2M3 has 3 mantissa bits to E3M2's 2.
    """
    rng = np.random.default_rng(0)
    weight = (rng.standard_normal((64, 256)) * 0.05).astype(np.float32)

    def rel_error(fmt: FP6Format) -> float:
        packed, scales = quantize_mxfp6(weight, fmt)
        decoded = dequantize_mxfp6(packed, scales, fmt)
        return float(np.linalg.norm(decoded - weight) / np.linalg.norm(weight))

    assert rel_error(FP6Format.E2M3) < rel_error(FP6Format.E3M2)


@ALL_FORMATS
def test_quantize_rejects_unaligned_k(fmt: FP6Format) -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        quantize_mxfp6(np.zeros((2, 48), np.float32), fmt)


@ALL_FORMATS
def test_quantize_rejects_non_finite(fmt: FP6Format) -> None:
    """FP6 has no Inf or NaN encoding, so a non-finite input must not silently
    saturate into a finite number."""
    weight = np.zeros((1, 32), np.float32)
    weight[0, 0] = np.inf
    with pytest.raises(ValueError, match="not finite"):
        quantize_mxfp6(weight, fmt)


@ALL_FORMATS
def test_encode_breaks_exact_ties_toward_even(fmt: FP6Format) -> None:
    """A value exactly between two codes must round to the even one.

    This is the property that separates round-to-nearest-even from
    round-half-away-from-zero, and it is invisible to a table comparison: both
    modes agree on every representable value and differ only at the midpoints.
    Getting it wrong biases every block away from zero.

    Code parity is the criterion because a code is
    ``(exponent << mantissa_width) | mantissa``, so its low bit *is* the
    mantissa's low bit -- including across a binade boundary, where the
    mantissa wraps to zero and the exponent carries.
    """
    table = _REFERENCE[fmt]
    for code in range(len(table) - 1):
        lo, hi = table[code], table[code + 1]
        midpoint = (lo + hi) / 2.0
        # Exactly one of two consecutive codes is even.
        expected = code if code % 2 == 0 else code + 1

        got = encode_f32_to_fp6(np.array([midpoint], np.float32), fmt)
        assert got[0] == expected, (
            f"{fmt.value}: midpoint {midpoint} of codes {code}/{code + 1}"
            f" encoded to {got[0]}, expected the even code {expected}"
        )

        # Same tie, negative side: sign must not change which way it breaks.
        got_neg = encode_f32_to_fp6(np.array([-midpoint], np.float32), fmt)
        assert got_neg[0] == expected | 0x20, (
            f"{fmt.value}: midpoint {-midpoint} encoded to {got_neg[0]},"
            f" expected {expected | 0x20}"
        )


@ALL_FORMATS
def test_encode_rounds_away_from_ties_normally(fmt: FP6Format) -> None:
    """Off-midpoint values still round to the nearer code, either side."""
    table = _REFERENCE[fmt]
    for code in range(len(table) - 1):
        lo, hi = table[code], table[code + 1]
        step = hi - lo
        assert (
            encode_f32_to_fp6(np.array([lo + step * 0.4], np.float32), fmt)[0]
            == code
        )
        assert encode_f32_to_fp6(np.array([lo + step * 0.6], np.float32), fmt)[
            0
        ] == (code + 1)


@ALL_FORMATS
def test_block_scale_uses_even_mode_not_floor(fmt: FP6Format) -> None:
    """The E8M0 scale must round the block max, not floor it.

    Flooring systematically underestimates a block's range -- the whole block
    then quantizes against a scale one binade too small, biasing every element
    low. The two modes differ only just below a power of two, where rounding at
    the FP6 mantissa boundary carries into the exponent and flooring does not.
    """
    # Just below 4.0, by less than half an FP6 mantissa step, so rounding at
    # the mantissa boundary carries into the next exponent.
    just_under = np.array([np.nextafter(np.float32(4.0), np.float32(0.0))])
    just_under = just_under.astype(np.float32)

    even = compute_mx_even_scale(just_under, fmt)
    floored = (just_under.view(np.uint32) >> np.uint32(23)).astype(
        np.int32
    ) - fmt.max_exponent

    assert even[0] == floored[0] + 1, (
        f"{fmt.value}: even-mode scale {even[0]} should be one binade above"
        f" the floored exponent {floored[0]} for a block max just under 4.0"
    )


@ALL_FORMATS
def test_block_scale_matches_floor_away_from_boundaries(
    fmt: FP6Format,
) -> None:
    """Away from a power-of-two boundary, rounding and flooring agree.

    Pins that even-mode is a rounding correction at the boundary and not a
    blanket exponent bump, which would waste a binade on every block.
    """
    mid = np.array([3.0], np.float32)  # squarely inside its binade
    even = compute_mx_even_scale(mid, fmt)
    floored = (mid.view(np.uint32) >> np.uint32(23)).astype(
        np.int32
    ) - fmt.max_exponent
    assert even[0] == floored[0]
