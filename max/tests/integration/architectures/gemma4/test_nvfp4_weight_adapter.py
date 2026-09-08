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

"""Weight-adapter tests for Gemma 4 NVFP4 checkpoints (no network, no GPU).

compressed-tensors NVFP4 exports (e.g. ``RedHatAI/gemma-4-31B-it-NVFP4``) store
the same block-scaled E2M1 weights as the modelopt export, but under different
tensor names and with the per-tensor global scales shaped ``[1]`` instead of
scalar. ``convert_safetensor_language_state_dict`` must reconcile those onto the
modelopt names the quantized ``Linear`` registers (``weight`` /
``weight_scale_2`` / ``input_scale``), while leaving genuine modelopt
checkpoints untouched. Shapes/dtypes below mirror layer 0 of
``RedHatAI/gemma-4-31B-it-NVFP4`` and ``nvidia/Gemma-4-31B-IT-NVFP4``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from max.pipelines.architectures.gemma4.weight_adapters import (
    convert_safetensor_language_state_dict,
)


class _FakeWeight:
    """Minimal stand-in exposing the ``.data()`` the adapter consumes."""

    def __init__(self, data: WeightData) -> None:
        self._data = data

    def data(self) -> WeightData:
        return self._data


def _wd(array: np.ndarray, name: str, dtype: DType) -> WeightData:
    return WeightData(array, name, dtype, Shape(list(array.shape)))


_PREFIX = "model.language_model.layers.0.mlp.gate_proj."


def test_compressed_tensors_names_remapped_to_modelopt() -> None:
    state_dict = {
        _PREFIX + "weight_packed": _FakeWeight(
            _wd(
                np.zeros((8, 4), np.uint8),
                _PREFIX + "weight_packed",
                DType.uint8,
            )
        ),
        _PREFIX + "weight_scale": _FakeWeight(
            _wd(
                np.zeros((8, 1), np.uint8),
                _PREFIX + "weight_scale",
                DType.float8_e4m3fn,
            )
        ),
        # Distinct, non-unit values so the reciprocal is observable.
        _PREFIX + "weight_global_scale": _FakeWeight(
            _wd(
                np.array([8.0], np.float32),
                _PREFIX + "weight_global_scale",
                DType.float32,
            )
        ),
        _PREFIX + "input_global_scale": _FakeWeight(
            _wd(
                np.array([4.0], np.float32),
                _PREFIX + "input_global_scale",
                DType.float32,
            )
        ),
    }

    out = convert_safetensor_language_state_dict(
        cast("dict[str, Weights]", state_dict)
    )
    base = "layers.0.mlp.gate_proj."

    # Packed weight + both global scales are renamed onto the modelopt names.
    assert set(out) == {
        base + "weight",
        base + "weight_scale",
        base + "weight_scale_2",
        base + "input_scale",
    }
    # No compressed-tensors names survive.
    assert not any(
        s in k
        for k in out
        for s in ("weight_packed", "weight_global_scale", "input_global_scale")
    )
    # Block scale passes through untouched; packed weight keeps its shape.
    assert tuple(out[base + "weight"].shape) == (8, 4)
    assert tuple(out[base + "weight_scale"].shape) == (8, 1)
    # Per-tensor global scales are inverted to the modelopt convention and
    # squeezed [1] -> scalar to match the quantized Linear.
    assert tuple(out[base + "weight_scale_2"].shape) == ()
    assert tuple(out[base + "input_scale"].shape) == ()
    assert float(np.from_dlpack(out[base + "weight_scale_2"].data)) == 0.125
    assert float(np.from_dlpack(out[base + "input_scale"].data)) == 0.25


def test_modelopt_names_pass_through_unchanged() -> None:
    state_dict = {
        _PREFIX + "weight": _FakeWeight(
            _wd(np.zeros((8, 4), np.uint8), _PREFIX + "weight", DType.uint8)
        ),
        _PREFIX + "weight_scale": _FakeWeight(
            _wd(
                np.zeros((8, 1), np.uint8),
                _PREFIX + "weight_scale",
                DType.float8_e4m3fn,
            )
        ),
        _PREFIX + "weight_scale_2": _FakeWeight(
            _wd(
                np.ones((), np.float32),
                _PREFIX + "weight_scale_2",
                DType.float32,
            )
        ),
        _PREFIX + "input_scale": _FakeWeight(
            _wd(np.ones((), np.float32), _PREFIX + "input_scale", DType.float32)
        ),
    }

    out = convert_safetensor_language_state_dict(
        cast("dict[str, Weights]", state_dict)
    )
    base = "layers.0.mlp.gate_proj."

    assert set(out) == {
        base + "weight",
        base + "weight_scale",
        base + "weight_scale_2",
        base + "input_scale",
    }
    # Scalars stay scalar — the [1]->() squeeze must not fire on modelopt.
    assert tuple(out[base + "weight_scale_2"].shape) == ()
    assert tuple(out[base + "input_scale"].shape) == ()
