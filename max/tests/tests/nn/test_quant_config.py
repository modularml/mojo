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

"""Checks for the MX-format dispatch helpers on ``QuantConfig``.

``is_mx``, ``mxfp6_format``, ``mx_element_dtype``, and ``fp4_packed_k`` are
the CPU-testable surface that decides which packing arithmetic a quantized
weight uses. MXFP4 and MXFP6 both travel as ``DType.uint8``, so a wrong
branch here silently mis-sizes a tensor rather than raising -- these run
without a GPU and don't need the lazy-trace machinery.
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
    fp4_packed_k,
)

_BLOCK_SIZE = (1, 32)


def _config(
    format: QuantFormat, mxfp6_element_format: str = "e2m3"
) -> QuantConfig:
    spec = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.DYNAMIC,
        dtype=DType.float32,
        block_size=_BLOCK_SIZE,
    )
    return QuantConfig(
        input_scale=spec,
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=_BLOCK_SIZE,
        ),
        mlp_quantized_layers={0},
        attn_quantized_layers={0},
        format=format,
        _mxfp6_element_format=mxfp6_element_format,
    )


@pytest.mark.parametrize(
    "format, expected",
    [
        (QuantFormat.MXFP4, True),
        (QuantFormat.MXFP6, True),
        (QuantFormat.MXFP8, True),
        (QuantFormat.NVFP4, False),
        (QuantFormat.BLOCKSCALED_FP8, False),
        (QuantFormat.INT8_W8A8, False),
    ],
)
def test_is_mx_covers_only_ocp_microscaling_formats(
    format: QuantFormat, expected: bool
) -> None:
    """NVFP4 shares the FP4 element type with MXFP4 but scales differently."""
    assert _config(format).is_mx is expected


def test_mxfp6_format_returns_the_configured_element_format() -> None:
    assert _config(QuantFormat.MXFP6, "e3m2").mxfp6_format == "e3m2"
    assert _config(QuantFormat.MXFP6, "e2m3").mxfp6_format == "e2m3"


def test_mxfp6_format_rejects_non_mxfp6_configs() -> None:
    with pytest.raises(ValueError, match="not an MXFP6 config"):
        _ = _config(QuantFormat.MXFP4).mxfp6_format


@pytest.mark.parametrize(
    "format, mxfp6_element_format, expected_dtype",
    [
        (QuantFormat.MXFP6, "e2m3", DType.float6_e2m3fn),
        (QuantFormat.MXFP6, "e3m2", DType.float6_e3m2fn),
        (QuantFormat.MXFP4, "e2m3", DType.float4_e2m1fn),
        (QuantFormat.MXFP8, "e2m3", DType.float8_e4m3fn),
    ],
)
def test_mx_element_dtype_distinguishes_packed_formats(
    format: QuantFormat, mxfp6_element_format: str, expected_dtype: DType
) -> None:
    """MXFP4 and MXFP6 both pack into ``uint8``; the element dtype disambiguates."""
    assert (
        _config(format, mxfp6_element_format).mx_element_dtype == expected_dtype
    )


def test_mx_element_dtype_rejects_non_mx_configs() -> None:
    with pytest.raises(ValueError, match="not an MX config"):
        _ = _config(QuantFormat.NVFP4).mx_element_dtype


def test_fp4_packed_k_returns_in_dim_for_unquantized() -> None:
    assert fp4_packed_k(256, None) == 256


@pytest.mark.parametrize(
    "format, in_dim, expected_packed_k",
    [
        # NVFP4/MXFP4 pack two codes per byte.
        (QuantFormat.NVFP4, 256, 128),
        (QuantFormat.MXFP4, 256, 128),
        # MXFP6 packs four codes per three bytes.
        (QuantFormat.MXFP6, 256, 192),
        # Whole-byte formats pass through unchanged.
        (QuantFormat.BLOCKSCALED_FP8, 256, 256),
        (QuantFormat.MXFP8, 256, 256),
    ],
)
def test_fp4_packed_k_matches_the_format_pack_ratio(
    format: QuantFormat, in_dim: int, expected_packed_k: int
) -> None:
    assert fp4_packed_k(in_dim, _config(format)) == expected_packed_k
