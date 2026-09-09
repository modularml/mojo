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

from __future__ import annotations

from unittest.mock import NonCallableMock

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData
from max.pipelines.architectures.laguna.weight_adapters import (
    convert_safetensor_state_dict,
)


def _mock_weight(data: WeightData | None = None) -> NonCallableMock:
    """Builds a mock ``Weights`` whose ``.data()`` returns ``data``.

    When ``data`` is None the returned value is an opaque mock, which is fine
    for keys whose tensor contents the adapter never inspects (structural
    renames). Scale keys need real ``WeightData`` so the reciprocation /
    dtype-reinterpretation paths can run.
    """
    weight = NonCallableMock()
    weight.data.return_value = data if data is not None else NonCallableMock()
    return weight


def test_convert_safetensor_state_dict() -> None:
    # Global scales are stored large by compressed-tensors (~5504); the adapter
    # must reciprocate them to MAX's multiplicative dequant convention.
    weight_global_scale = WeightData.from_numpy(
        np.array(5504.0, dtype=np.float32), "wgs"
    )
    input_global_scale = WeightData.from_numpy(
        np.array(1184.0, dtype=np.float32), "igs"
    )
    # Per-group e4m3 scale shipped as raw uint8 — reinterpreted, not rescaled.
    weight_scale = WeightData.from_numpy(np.zeros((8, 4), dtype=np.uint8), "ws")

    state_dict = {
        "model.layers.0.self_attn.q_proj.weight": _mock_weight(),
        "model.layers.0.mlp.experts.gate_proj.weight_packed": _mock_weight(),
        "model.layers.0.mlp.experts.gate_proj.weight_scale": _mock_weight(
            weight_scale
        ),
        "model.layers.0.mlp.experts.gate_proj.weight_global_scale": (
            _mock_weight(weight_global_scale)
        ),
        "model.layers.0.mlp.experts.gate_proj.input_global_scale": (
            _mock_weight(input_global_scale)
        ),
        "model.layers.0.mlp.gate.weight": _mock_weight(),
        "model.layers.0.mlp.experts.e_score_correction_bias": _mock_weight(),
        "model.layers.0.mlp.shared_expert.gate_proj.weight": _mock_weight(),
        # FP8 KV-cache static scales: MAX scales KV dynamically, so these are
        # dropped rather than loaded.
        "model.layers.0.self_attn.k_scale": _mock_weight(),
        "model.layers.0.self_attn.v_scale": _mock_weight(),
    }

    out = convert_safetensor_state_dict(
        state_dict,  # type: ignore[arg-type]
        NonCallableMock(),
        NonCallableMock(),
    )

    # ``model.`` prefix stripped; structural renames applied.
    assert "layers.0.self_attn.q_proj.weight" in out
    assert "layers.0.mlp.experts.gate_proj.weight" in out  # weight_packed
    assert "layers.0.mlp.gate.gate_score.weight" in out  # gate.weight
    assert "layers.0.mlp.gate.e_score_correction_bias" in out
    assert "layers.0.mlp.shared_experts.gate_proj.weight" in out

    # FP8 KV-cache static scales dropped.
    assert not any(k.endswith(".k_scale") for k in out)
    assert not any(k.endswith(".v_scale") for k in out)

    # Global scales renamed AND reciprocated to the multiplicative convention.
    wgs = out["layers.0.mlp.experts.gate_proj.weight_scale_2"]
    igs = out["layers.0.mlp.experts.gate_proj.input_scale"]
    np.testing.assert_allclose(np.from_dlpack(wgs), 1.0 / 5504.0, rtol=1e-6)
    np.testing.assert_allclose(np.from_dlpack(igs), 1.0 / 1184.0, rtol=1e-6)

    # Per-group uint8 weight_scale reinterpreted as float8_e4m3fn (not rescaled).
    ws = out["layers.0.mlp.experts.gate_proj.weight_scale"]
    assert ws.dtype == DType.float8_e4m3fn
