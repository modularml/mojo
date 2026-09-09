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

"""CPU-only checks for the FLUX.2 int8 W8A8 weight/Linear reconciliation.

``verify_int8_quantization_consistency`` guards against
``_INT8_QUANTIZED_WEIGHT_SUFFIXES`` (which weights get RTN-quantized to int8)
drifting out of sync with which Linears ``Flux2BlockQuant.resolve`` configures
as int8. Without it the drift surfaces far away as ``weight b must be int8,
got bf16`` from the matmul op at graph build. These tests cover the match case
and both mismatch directions, asserting the error names the offending layer.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.shape import Shape
from max.graph.weights import WeightData
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.pipelines.architectures.flux2.weight_adapters import (
    verify_int8_quantization_consistency,
)

_DREF = DeviceRef.CPU()


def _int8_quant_config() -> QuantConfig:
    """A symmetric int8 W8A8 ``QuantConfig`` (matches FLUX.2-klein's)."""
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.COLWISE,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.ROWWISE,
            dtype=DType.float32,
        ),
        mlp_quantized_layers={0},
        attn_quantized_layers={0},
        format=QuantFormat.INT8_W8A8,
    )


class _TwoLinear(Module):
    """A tiny module with two Linears, each optionally int8-configured."""

    def __init__(self, a_int8: bool, b_int8: bool) -> None:
        super().__init__()
        qc = _int8_quant_config()
        self.a = Linear(
            in_dim=8,
            out_dim=8,
            dtype=DType.bfloat16,
            device=_DREF,
            quant_config=qc if a_int8 else None,
        )
        self.b = Linear(
            in_dim=8,
            out_dim=8,
            dtype=DType.bfloat16,
            device=_DREF,
            quant_config=qc if b_int8 else None,
        )

    def __call__(self, x):  # noqa: ANN001 - unused, Module requires it
        return x


def _weight(name: str, dtype: DType) -> WeightData:
    arr = np.zeros(
        (8, 8), dtype=(np.int8 if dtype == DType.int8 else np.float32)
    )
    return WeightData(Buffer.from_numpy(arr), name, dtype, Shape((8, 8)))


def test_consistency_ok_when_int8_config_matches_int8_weights() -> None:
    """No raise when every int8-configured Linear has an int8 weight."""
    model = _TwoLinear(a_int8=True, b_int8=True)
    state_dict = {
        "a.weight": _weight("a.weight", DType.int8),
        "b.weight": _weight("b.weight", DType.int8),
    }
    verify_int8_quantization_consistency(model, state_dict)


def test_consistency_ok_when_no_int8_anywhere() -> None:
    """No raise when nothing is int8 (bf16 model + bf16 weights)."""
    model = _TwoLinear(a_int8=False, b_int8=False)
    state_dict = {
        "a.weight": _weight("a.weight", DType.bfloat16),
        "b.weight": _weight("b.weight", DType.bfloat16),
    }
    verify_int8_quantization_consistency(model, state_dict)


def test_consistency_raises_when_configured_int8_but_weight_bf16() -> None:
    """A Linear configured int8 whose weight stayed bf16 (whitelist too narrow)."""
    model = _TwoLinear(a_int8=True, b_int8=True)
    state_dict = {
        "a.weight": _weight("a.weight", DType.int8),
        "b.weight": _weight("b.weight", DType.bfloat16),  # drift
    }
    with pytest.raises(ValueError, match=r"b\.weight") as exc:
        verify_int8_quantization_consistency(model, state_dict)
    assert "configured int8 but weight left bf16" in str(exc.value)


def test_consistency_raises_when_weight_int8_but_linear_bf16() -> None:
    """A weight quantized to int8 whose Linear is bf16 (whitelist too broad)."""
    model = _TwoLinear(a_int8=True, b_int8=False)
    state_dict = {
        "a.weight": _weight("a.weight", DType.int8),
        "b.weight": _weight("b.weight", DType.int8),  # drift
    }
    with pytest.raises(ValueError, match=r"b\.weight") as exc:
        verify_int8_quantization_consistency(model, state_dict)
    assert "quantized to int8 but Linear is bf16" in str(exc.value)
