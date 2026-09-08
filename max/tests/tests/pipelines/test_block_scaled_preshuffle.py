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
"""Tests for the MX expert-weight preshuffle helpers (KERN-3393).

The float8 cases back each ``WeightData`` with a MAX ``Buffer`` whose DLPack
dtype is a float8 numpy cannot import — the exact shape of the production
state dict. numpy's ``from_dlpack`` error type for those changed from
``RuntimeError`` to ``BufferError`` in numpy 2.5.0 (numpy gh-30937); the
helpers used to dispatch on that type and, under numpy >= 2.5, silently
skipped every MXFP8 expert while the caller still flipped
``block_scaled_preshuffled_b`` — serving row-major weights to the preb kernel.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.pipelines.weights.block_scaled_preshuffle import (
    preshuffle_block_scaled_b_experts,
    preshuffle_block_scaled_b_scales,
)

_N, _K_BYTES = 16, 64
_MN, _K_SCALES = 32, 8


def _weight_bytes(seed: int, shape: tuple[int, int]) -> np.ndarray:
    return (
        np.random.default_rng(seed)
        .integers(0, 256, size=shape)
        .astype(np.uint8)
    )


def _f8_weight_data(name: str, raw: np.ndarray, dtype: DType) -> WeightData:
    """WeightData backed by a MAX Buffer whose DLPack dtype is ``dtype``."""
    buf = Buffer.from_dlpack(raw).view(dtype, raw.shape)
    return WeightData(buf, name, dtype, Shape(raw.shape))


def _expected_b_5d(src: np.ndarray) -> np.ndarray:
    n, k_bytes = src.shape
    return (
        src.reshape(n // 16, 16, k_bytes // 64, 4, 16)
        .transpose(0, 2, 3, 1, 4)
        .reshape(n, k_bytes)
    )


def _expected_scale_4d(src: np.ndarray) -> np.ndarray:
    mn, k_scales = src.shape
    return (
        src.reshape(mn // 32, 2, 16, k_scales // 8, 2, 4)
        .transpose(0, 3, 5, 2, 4, 1)
        .reshape(mn, k_scales)
    )


def _result_bytes(wd: WeightData) -> np.ndarray:
    return np.from_dlpack(
        wd.to_buffer().view(DType.uint8, wd.shape.static_dims)
    )


def test_preshuffle_b_experts_float8_buffer_backed() -> None:
    """MXFP8 experts behind a float8 DLPack producer must be preshuffled."""
    names = [
        f"layers.0.mlp.experts.{i}.{proj}.weight"
        for i in range(2)
        for proj in ("gate_proj", "up_proj", "down_proj")
    ]
    raws = {n: _weight_bytes(i, (_N, _K_BYTES)) for i, n in enumerate(names)}
    state_dict = {
        n: _f8_weight_data(n, raw, DType.float8_e4m3fn)
        for n, raw in raws.items()
    }

    preshuffle_block_scaled_b_experts(state_dict)

    for n, raw in raws.items():
        assert state_dict[n].dtype == DType.float8_e4m3fn
        np.testing.assert_array_equal(
            _result_bytes(state_dict[n]), _expected_b_5d(raw)
        )


def test_preshuffle_b_experts_uint8() -> None:
    """MXFP4-packed uint8 experts (Kimi K2.5 path) keep working."""
    name = "language_model.layers.3.mlp.experts.7.up_proj.weight"
    raw = _weight_bytes(7, (_N, _K_BYTES))
    state_dict = {name: WeightData.from_numpy(raw.copy(), name)}

    preshuffle_block_scaled_b_experts(state_dict)

    assert state_dict[name].dtype == DType.uint8
    np.testing.assert_array_equal(
        _result_bytes(state_dict[name]), _expected_b_5d(raw)
    )


def test_preshuffle_b_scales_e8m0_buffer_backed() -> None:
    """E8M0 scales behind a float8 DLPack producer must be preshuffled."""
    name = "layers.0.mlp.experts.0.gate_proj.weight_scale"
    raw = _weight_bytes(11, (_MN, _K_SCALES))
    state_dict = {name: _f8_weight_data(name, raw, DType.float8_e8m0fnu)}

    preshuffle_block_scaled_b_scales(state_dict)

    assert state_dict[name].dtype == DType.float8_e8m0fnu
    np.testing.assert_array_equal(
        _result_bytes(state_dict[name]), _expected_scale_4d(raw)
    )


def test_preshuffle_b_experts_rejects_unshuffleable_group() -> None:
    """A matched group with no shuffleable weight raises instead of no-oping."""
    name = "layers.0.mlp.experts.0.gate_proj.weight"
    raw = np.zeros((_N, _K_BYTES // 2), dtype=np.float32)
    state_dict = {name: WeightData.from_numpy(raw, name)}

    with pytest.raises(ValueError, match="preshuffle skipped"):
        preshuffle_block_scaled_b_experts(state_dict)


def test_preshuffle_b_experts_rejects_partial_group() -> None:
    """A group mixing shuffleable and unshuffleable weights raises."""
    good = "layers.0.mlp.experts.0.gate_proj.weight"
    bad = "layers.0.mlp.experts.1.gate_proj.weight"
    state_dict = {
        good: WeightData.from_numpy(_weight_bytes(0, (_N, _K_BYTES)), good),
        bad: WeightData.from_numpy(
            np.zeros((_N, _K_BYTES), dtype=np.float32), bad
        ),
    }

    with pytest.raises(ValueError, match="preshuffle skipped"):
        preshuffle_block_scaled_b_experts(state_dict)


def test_preshuffle_b_scales_rejects_unshuffleable_group() -> None:
    """A matched scale group with a non-E8M0 scale raises."""
    name = "layers.0.mlp.experts.0.gate_proj.weight_scale"
    raw = np.zeros((_MN, _K_SCALES), dtype=np.float32)
    state_dict = {name: WeightData.from_numpy(raw, name)}

    with pytest.raises(ValueError, match="preshuffle skipped"):
        preshuffle_block_scaled_b_scales(state_dict)


def test_preshuffle_b_experts_counts_matches_under_virtual_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Virtual devices skip the byte copy, so identity cannot report a match.

    The graph-dump tools compile against virtual devices only. Reporting 0
    there would make a clean dump indistinguishable from a naming regression.
    """
    monkeypatch.setattr(
        "max.pipelines.weights.block_scaled_preshuffle.is_virtual_device_mode",
        lambda: True,
    )
    name = "layers.0.mlp.experts.0.gate_proj.weight"
    state_dict = {
        name: WeightData.from_numpy(_weight_bytes(3, (_N, _K_BYTES)), name)
    }
    before = state_dict[name]

    assert preshuffle_block_scaled_b_experts(state_dict) == 1
    assert state_dict[name] is before, "virtual mode must not permute bytes"
