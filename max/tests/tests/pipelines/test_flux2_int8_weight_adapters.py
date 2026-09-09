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

"""CPU-only checks for the FLUX.2 int8 W8A8 RTN weight quantization.

Covers both checkpoint sources of the int8 W8A8 path:

- bf16 -> int8 (``_rtn_quantize_int8``): regression-pins the rowwise
  absmax/127 quantization against an independent numpy reference.
- NVFP4 -> int8 (``_rtn_quantize_int8_from_nvfp4``): a synthetic BFL-named
  NVFP4 checkpoint (packed E2M1 nibbles + fp8-e4m3fn block scales +
  per-tensor ``weight_scale_2``) round-trips through the real conversion
  (nibble swap, naming, stacked-QKV split) and requantizes to int8 whose
  dequant reproduces an independently reconstructed fp32 weight within the
  per-channel RTN rounding bound.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph.shape import Shape
from max.graph.weights import WeightData
from max.pipelines.architectures.flux2.weight_adapters import (
    _f8e4m3fn_to_fp32,
    _nvfp4_weightdata_to_fp32,
    _rtn_quantize_int8,
    _rtn_quantize_int8_from_nvfp4,
)

# Independent E2M1 lookup table (bit 3 sign, bits 2:1 exponent, bit 0
# mantissa), written out separately from the implementation's table.
_REF_E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

# fp32 -> float8-e4m3fn bit patterns for a few exactly-representable scales
# (1 sign, 4 exponent bits with bias 7, 3 mantissa bits).
_F8E4M3_BITS = {0.5: 0x30, 1.0: 0x38, 1.5: 0x3C, 2.0: 0x40}

_BLOCK = 16  # NVFP4 K-axis block size.


def _bf16_weight(name: str, w_fp32: np.ndarray) -> WeightData:
    """Build a bf16 WeightData from fp32 values (truncating rounding)."""
    u16 = (w_fp32.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    buf = Buffer.from_numpy(np.ascontiguousarray(u16)).view(
        DType.bfloat16, w_fp32.shape
    )
    return WeightData(buf, name, DType.bfloat16, Shape(w_fp32.shape))


def _truncate_to_bf16(w_fp32: np.ndarray) -> np.ndarray:
    """Round fp32 values to their bf16-truncated fp32 equivalents."""
    u32 = w_fp32.astype(np.float32).view(np.uint32) & np.uint32(0xFFFF0000)
    return u32.view(np.float32)


def _pack_nibbles_hi_first(nibbles: np.ndarray) -> np.ndarray:
    """Pack logical E2M1 nibbles two-per-byte, first value in the HIGH
    nibble -- the BFL checkpoint convention that
    ``convert_nvfp4_state_dict`` swaps to lo-first at load."""
    return ((nibbles[:, 0::2] << 4) | nibbles[:, 1::2]).astype(np.uint8)


def _make_nvfp4_entries(
    bfl_name: str,
    nibbles: np.ndarray,
    scales_fp32: np.ndarray,
    weight_scale_2: float,
    input_scale: float,
) -> dict[str, WeightData]:
    """Build the four BFL NVFP4 entries for one Linear.

    ``nibbles`` is the logical ``[N, K]`` E2M1 code array; ``scales_fp32``
    the ``[N, K // 16]`` block scales, restricted to values in
    ``_F8E4M3_BITS``.
    """
    packed = _pack_nibbles_hi_first(nibbles)
    scale_bits = np.vectorize(lambda s: _F8E4M3_BITS[float(s)])(
        scales_fp32
    ).astype(np.uint8)
    ws2 = np.array([weight_scale_2], dtype=np.float32)
    in_s = np.array([input_scale], dtype=np.float32)
    return {
        f"{bfl_name}.weight": WeightData(
            Buffer.from_numpy(packed),
            f"{bfl_name}.weight",
            DType.uint8,
            Shape(packed.shape),
        ),
        f"{bfl_name}.weight_scale": WeightData(
            Buffer.from_numpy(scale_bits).view(
                DType.float8_e4m3fn, scale_bits.shape
            ),
            f"{bfl_name}.weight_scale",
            DType.float8_e4m3fn,
            Shape(scale_bits.shape),
        ),
        f"{bfl_name}.weight_scale_2": WeightData(
            Buffer.from_numpy(ws2),
            f"{bfl_name}.weight_scale_2",
            DType.float32,
            Shape(ws2.shape),
        ),
        f"{bfl_name}.input_scale": WeightData(
            Buffer.from_numpy(in_s),
            f"{bfl_name}.input_scale",
            DType.float32,
            Shape(in_s.shape),
        ),
    }


def _ref_nvfp4_to_fp32(
    nibbles: np.ndarray, scales_fp32: np.ndarray, weight_scale_2: float
) -> np.ndarray:
    """Independent NVFP4 reconstruction: LUT * block scale * tensor scale."""
    return (
        _REF_E2M1[nibbles]
        * np.repeat(scales_fp32.astype(np.float32), _BLOCK, axis=1)
        * np.float32(weight_scale_2)
    )


def _assert_int8_rtn_matches(
    out: dict[str, WeightData], key: str, ref_fp32: np.ndarray
) -> None:
    """Assert ``out[key]`` is the rowwise-RTN int8 quantization of
    ``ref_fp32``: exact absmax/127 scales and dequant within scale/2."""
    scale_key = key[: -len(".weight")] + ".weight_scale"
    q = out[key]
    s = out[scale_key]
    assert q.dtype == DType.int8
    assert tuple(int(d) for d in q.shape) == ref_fp32.shape
    assert s.dtype == DType.float32
    assert tuple(int(d) for d in s.shape) == (ref_fp32.shape[0], 1)

    scale = s.to_buffer().to_numpy()
    absmax = np.abs(ref_fp32).max(axis=1, keepdims=True)
    expected_scale = np.where(
        absmax != 0.0, absmax / 127.0, np.float32(1.0)
    ).astype(np.float32)
    np.testing.assert_array_equal(scale, expected_scale)

    # RTN bounds the error at scale/2 (elements can land exactly on the .5
    # rounding boundary); allow fp32 evaluation headroom past the bound.
    dequant = q.to_buffer().to_numpy().astype(np.float32) * scale
    assert np.all(np.abs(dequant - ref_fp32) <= scale / 2 + 1e-6)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


def test_e2m1_decode_matches_independent_lut() -> None:
    """All 16 E2M1 codes decode to the independent LUT values."""
    nibbles = np.arange(16, dtype=np.uint8).reshape(1, 16)
    # _nvfp4_weightdata_to_fp32 consumes lo-first packing (post-swap).
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).astype(np.uint8)
    weight = WeightData(
        Buffer.from_numpy(packed), "w.weight", DType.uint8, Shape(packed.shape)
    )
    unit_bits = np.full((1, 1), _F8E4M3_BITS[1.0], dtype=np.uint8)
    scale = WeightData(
        Buffer.from_numpy(unit_bits).view(DType.float8_e4m3fn, (1, 1)),
        "w.weight_scale",
        DType.float8_e4m3fn,
        Shape((1, 1)),
    )
    one = np.array([1.0], dtype=np.float32)
    ws2 = WeightData(
        Buffer.from_numpy(one), "w.weight_scale_2", DType.float32, Shape((1,))
    )
    decoded = _nvfp4_weightdata_to_fp32(weight, scale, ws2)
    np.testing.assert_array_equal(decoded, _REF_E2M1.reshape(1, 16))


def test_f8e4m3fn_decode_known_values() -> None:
    """fp8-e4m3fn decode: normals, subnormals, signs, and the max finite."""
    cases = {
        0x00: 0.0,
        0x30: 0.5,
        0x38: 1.0,
        0x3C: 1.5,
        0x40: 2.0,
        0xB8: -1.0,
        0x01: 2.0**-9,  # smallest subnormal
        0x07: 7.0 * 2.0**-9,  # largest subnormal
        0x7E: 448.0,  # max finite
    }
    bits = np.array(list(cases.keys()), dtype=np.uint8)
    value = WeightData(
        Buffer.from_numpy(bits).view(DType.float8_e4m3fn, bits.shape),
        "s.weight_scale",
        DType.float8_e4m3fn,
        Shape(bits.shape),
    )
    np.testing.assert_array_equal(
        _f8e4m3fn_to_fp32(value),
        np.array(list(cases.values()), dtype=np.float32),
    )


def test_rtn_quantize_int8_bf16_reference(rng: np.random.Generator) -> None:
    """The bf16 -> int8 path matches the independent rowwise reference.

    Regression-pins ``_rtn_quantize_int8`` across the shared-helper
    refactor: same int8 values, same fp32 [N, 1] scales, same passthrough.
    """
    w = _truncate_to_bf16(rng.standard_normal((4, 32), dtype=np.float32) * 0.05)
    norm = _truncate_to_bf16(rng.standard_normal(8, dtype=np.float32))
    state_dict = {
        "transformer_blocks.0.attn.to_q.weight": _bf16_weight(
            "transformer_blocks.0.attn.to_q.weight", w
        ),
        "transformer_blocks.0.attn.norm_q.weight": _bf16_weight(
            "transformer_blocks.0.attn.norm_q.weight", norm
        ),
    }
    out = _rtn_quantize_int8(state_dict)

    _assert_int8_rtn_matches(out, "transformer_blocks.0.attn.to_q.weight", w)
    # Exact int8 codes per the reference formula (rint of w / scale).
    scale = (
        out["transformer_blocks.0.attn.to_q.weight_scale"]
        .to_buffer()
        .to_numpy()
    )
    expected_q = np.clip(np.rint(w / scale).astype(np.int32), -127, 127).astype(
        np.int8
    )
    np.testing.assert_array_equal(
        out["transformer_blocks.0.attn.to_q.weight"].to_buffer().to_numpy(),
        expected_q,
    )
    # Non-targeted weights pass through untouched.
    passthrough = out["transformer_blocks.0.attn.norm_q.weight"]
    assert passthrough.dtype == DType.bfloat16
    assert passthrough is state_dict["transformer_blocks.0.attn.norm_q.weight"]


def test_rtn_quantize_int8_from_nvfp4(rng: np.random.Generator) -> None:
    """A synthetic BFL NVFP4 checkpoint requantizes to int8 W8A8.

    Exercises the full path: BFL -> diffusers naming, hi-first nibble swap,
    stacked-QKV split, FP4 + two-level-scale reconstruction, rowwise int8
    RTN, NVFP4 side-tensor cleanup, and the bf16 RTN route for targeted
    Linears the checkpoint left unquantized.
    """
    n, k = 4, 32
    qkv_nibbles = rng.integers(0, 16, size=(3 * n, k), dtype=np.uint8)
    qkv_scales = rng.choice(
        np.array(list(_F8E4M3_BITS), dtype=np.float32),
        size=(3 * n, k // _BLOCK),
    )
    mlp_nibbles = rng.integers(0, 16, size=(n, k), dtype=np.uint8)
    mlp_scales = rng.choice(
        np.array(list(_F8E4M3_BITS), dtype=np.float32),
        size=(n, k // _BLOCK),
    )
    txt_proj = _truncate_to_bf16(
        rng.standard_normal((n, k), dtype=np.float32) * 0.02
    )
    norm = _truncate_to_bf16(rng.standard_normal(8, dtype=np.float32))

    state_dict = {
        **_make_nvfp4_entries(
            "double_blocks.0.img_attn.qkv", qkv_nibbles, qkv_scales, 0.03, 0.9
        ),
        **_make_nvfp4_entries(
            "double_blocks.0.img_mlp.0", mlp_nibbles, mlp_scales, 0.05, 1.1
        ),
        # A targeted Linear the checkpoint left in bf16 (absent from its
        # quantization metadata): must take the bf16 RTN route to int8.
        "double_blocks.0.txt_attn.proj.weight": _bf16_weight(
            "double_blocks.0.txt_attn.proj.weight", txt_proj
        ),
        # Non-targeted bf16 passthrough.
        "double_blocks.0.img_attn.norm.query_norm.scale": _bf16_weight(
            "double_blocks.0.img_attn.norm.query_norm.scale", norm
        ),
    }
    out = _rtn_quantize_int8_from_nvfp4(state_dict)

    # No NVFP4 side tensors survive the requant.
    assert not any(
        key.endswith((".weight_scale_2", ".input_scale")) for key in out
    )

    # Stacked QKV split into Q/K/V, each the RTN of its row block of the
    # independently reconstructed fp32 weight.
    qkv_ref = _ref_nvfp4_to_fp32(qkv_nibbles, qkv_scales, 0.03)
    for i, proj in enumerate(("to_q", "to_k", "to_v")):
        _assert_int8_rtn_matches(
            out,
            f"transformer_blocks.0.attn.{proj}.weight",
            qkv_ref[i * n : (i + 1) * n],
        )

    # Non-stacked NVFP4 Linear (BFL img_mlp.0 -> ff.linear_in).
    mlp_ref = _ref_nvfp4_to_fp32(mlp_nibbles, mlp_scales, 0.05)
    _assert_int8_rtn_matches(
        out, "transformer_blocks.0.ff.linear_in.weight", mlp_ref
    )

    # bf16-sourced targeted Linear also lands int8.
    _assert_int8_rtn_matches(
        out, "transformer_blocks.0.attn.to_add_out.weight", txt_proj
    )

    # Non-targeted weight passes through as bf16 under its diffusers name.
    passthrough = out["transformer_blocks.0.attn.norm_q.weight"]
    assert passthrough.dtype == DType.bfloat16
