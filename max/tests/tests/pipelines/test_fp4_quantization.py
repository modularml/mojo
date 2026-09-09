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
"""Tests the host-side NVFP4 encoder round trip."""

from __future__ import annotations

import numpy as np
import pytest
from max.pipelines.weights.fp4_quantization import (
    FP4Format,
    dequantize_nvfp4,
    e2m1_decode_table,
    encode_f32_to_e4m3,
    pack_pairs_e2m1_uint8,
    quantize_nvfp4,
    round_to_e2m1,
    unpack_e4m3_uint8_to_fp32,
    unpack_pairs_e2m1_to_fp32,
)

ALL_FORMATS = pytest.mark.parametrize("fmt", list(FP4Format))

E2M1_POSITIVE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@ALL_FORMATS
def test_decode_table_matches_e2m1(fmt: FP4Format) -> None:
    table = e2m1_decode_table(fmt)
    assert table.shape == (16,)
    np.testing.assert_array_equal(
        table[:8], np.array(E2M1_POSITIVE, dtype=np.float32)
    )
    np.testing.assert_array_equal(table[8:], -table[:8])


def test_e4m3_rounds_rather_than_truncates() -> None:
    """1.1 is closer to 1.125 than 1.0; bit-slicing would keep 1.0."""
    encoded = encode_f32_to_e4m3(np.array([1.1], dtype=np.float32))
    np.testing.assert_array_equal(
        unpack_e4m3_uint8_to_fp32(encoded), np.array([1.125], dtype=np.float32)
    )


def test_e4m3_subnormals_are_multiples_of_two_to_the_minus_nine() -> None:
    """exp=0, mant=k is k * 2^-9. Encode emits those codes for tiny scales."""
    codes = np.arange(8, dtype=np.uint8)
    expected = codes.astype(np.float32) * np.float32(2.0**-9)
    np.testing.assert_array_equal(unpack_e4m3_uint8_to_fp32(codes), expected)
    np.testing.assert_array_equal(encode_f32_to_e4m3(expected), codes)
    min_normal = unpack_e4m3_uint8_to_fp32(np.array([0x08], dtype=np.uint8))
    np.testing.assert_array_equal(min_normal, np.array([2.0**-6], np.float32))


def test_round_to_e2m1_nearest_even() -> None:
    """Midpoints land on the even code, matching ``cast_fp_to_fp4e2m1``."""
    midpoints = np.array(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32
    )
    expected = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0], dtype=np.float32)
    np.testing.assert_array_equal(round_to_e2m1(midpoints), expected)
    np.testing.assert_array_equal(round_to_e2m1(-midpoints), -expected)


@ALL_FORMATS
def test_pack_unpack_round_trip(fmt: FP4Format) -> None:
    table = e2m1_decode_table(fmt)
    packed = pack_pairs_e2m1_uint8(table)
    np.testing.assert_array_equal(unpack_pairs_e2m1_to_fp32(packed), table)


@ALL_FORMATS
def test_quantize_shapes(fmt: FP4Format) -> None:
    weight = np.zeros((8, 64), dtype=np.float32)
    packed, block_scales, decode_scale = quantize_nvfp4(weight, fmt)
    assert packed.shape == (8, 64 // 2)
    assert block_scales.shape == (8, 64 // fmt.block_size, 1)
    assert packed.dtype == block_scales.dtype == np.uint8
    assert np.isscalar(decode_scale)
    assert np.issubdtype(np.asarray(decode_scale).dtype, np.floating)


@ALL_FORMATS
def test_zero_block(fmt: FP4Format) -> None:
    packed, block_scales, decode_scale = quantize_nvfp4(
        np.zeros((1, fmt.block_size), np.float32), fmt
    )
    assert (
        dequantize_nvfp4(packed, block_scales, decode_scale, fmt) == 0
    ).all()


@ALL_FORMATS
def test_quantize_is_exact_for_representable_values(fmt: FP4Format) -> None:
    """A tensor already on the format's grid must quantize losslessly.

    NVFP4 derives one global scale from the tensor amax, so every element
    must share the same magnitude. Signs may vary.
    """
    rng = np.random.default_rng(0)
    table = e2m1_decode_table(fmt)
    magnitudes = np.unique(np.abs(table[table != 0]))
    magnitude = magnitudes[rng.integers(len(magnitudes))]
    candidates = table[np.abs(table) == magnitude]
    weight = rng.choice(candidates, (4, 64)).astype(np.float32)
    packed, block_scales, decode_scale = quantize_nvfp4(weight, fmt)
    np.testing.assert_array_equal(
        dequantize_nvfp4(packed, block_scales, decode_scale, fmt), weight
    )


@ALL_FORMATS
def test_quantize_roundtrip(fmt: FP4Format) -> None:
    rng = np.random.default_rng(0)
    weight = (rng.standard_normal((4, 32)) * 0.05).astype(np.float32)
    packed, block_scales, decode_scale = quantize_nvfp4(weight, fmt)
    decoded = dequantize_nvfp4(packed, block_scales, decode_scale, fmt)
    rel_error = float(np.linalg.norm(decoded - weight) / np.linalg.norm(weight))
    assert rel_error < 0.2
