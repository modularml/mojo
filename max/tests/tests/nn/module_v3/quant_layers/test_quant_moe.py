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

"""Lazy-trace tests for QuantizedMoE."""

from __future__ import annotations

from unittest.mock import MagicMock

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor, default_dtype
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.deepseekV3_modulev3.layers.quant_moe import (
    QuantizedMoE,
)
from max.pipelines.architectures.deepseekV3_modulev3.layers.quant_tensor import (
    FP8BlockTensor,
    NVFP4Tensor,
)

_HIDDEN_DIM = 256
_MOE_DIM = 128
_NUM_EXPERTS = 4
_NUM_EXPERTS_PER_TOKEN = 2
_SHARED_EXPERTS_DIM = 128
_SEQ_LEN = 4


_NVFP4_LEAVES = (
    "weight.data",
    "weight.weight_scale",
    "weight.weight_scale_2",
    "weight.input_scale",
)


def _expected_expert_param_names(
    quantized: bool, leaves: tuple[str, ...] | None = None
) -> set[str]:
    """Per-expert MLP parameter names for the routed experts."""
    if leaves is None:
        leaves = (
            ("weight.data", "weight.weight_scale_inv")
            if quantized
            else ("weight",)
        )
    names = set()
    for i in range(_NUM_EXPERTS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for leaf in leaves:
                names.add(f"experts.{i}.{proj}.{leaf}")
    return names


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


def test_moe_bf16_parameters(mock_accelerator: MagicMock) -> None:
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
        ).to(device)

        names = {name for name, _ in layer.parameters}
        expected = {"gate.gate_score.weight"} | _expected_expert_param_names(
            quantized=False
        )
        assert names == expected


def test_moe_fp8_parameters(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
            quant_config=fp8_quant_config,
        ).to(device)

        names = {name for name, _ in layer.parameters}
        # The gate is never quantized.
        expected = {"gate.gate_score.weight"} | _expected_expert_param_names(
            quantized=True
        )
        assert names == expected


def test_moe_shared_experts_parameters(mock_accelerator: MagicMock) -> None:
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
            has_shared_experts=True,
            shared_experts_dim=_SHARED_EXPERTS_DIM,
        ).to(device)

        names = {name for name, _ in layer.parameters}
        shared = {
            "shared_experts.gate_proj.weight",
            "shared_experts.up_proj.weight",
            "shared_experts.down_proj.weight",
        }
        assert shared <= names
        assert layer.shared_experts is not None


def test_moe_no_shared_experts_by_default(mock_accelerator: MagicMock) -> None:
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
        ).to(device)
        assert layer.shared_experts is None


# --------------------------------------------------------------------------- #
# Stacked expert weights
# --------------------------------------------------------------------------- #


def test_moe_stacked_weights_bf16(mock_accelerator: MagicMock) -> None:
    """bf16 stacked weights are plain tensors with a leading expert axis."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
        ).to(device)

        gate_up = layer.gate_up_proj
        assert len(gate_up) == 1
        assert isinstance(gate_up[0], Tensor)
        # gate || up stacked on the output axis: 2 * moe_dim.
        assert list(gate_up[0].shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM,
            _HIDDEN_DIM,
        ]

        down = layer.down_proj
        assert len(down) == 1
        assert isinstance(down[0], Tensor)
        assert list(down[0].shape) == [_NUM_EXPERTS, _HIDDEN_DIM, _MOE_DIM]


def test_moe_stacked_weights_fp8(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    """FP8 stacked weights are FP8BlockTensors with matching data shapes."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
            quant_config=fp8_quant_config,
        ).to(device)

        gate_up = layer.gate_up_proj
        assert len(gate_up) == 1
        assert isinstance(gate_up[0], FP8BlockTensor)
        assert gate_up[0].data.dtype == DType.float8_e4m3fn
        assert gate_up[0].weight_scale_inv.dtype == DType.float32
        assert list(gate_up[0].data.shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM,
            _HIDDEN_DIM,
        ]

        down = layer.down_proj
        assert len(down) == 1
        assert isinstance(down[0], FP8BlockTensor)
        assert list(down[0].data.shape) == [_NUM_EXPERTS, _HIDDEN_DIM, _MOE_DIM]


# --------------------------------------------------------------------------- #
# Forward
# --------------------------------------------------------------------------- #


def test_moe_bf16_forward(mock_accelerator: MagicMock) -> None:
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
        ).to(device)
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
        )
        out = layer(x)

        assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
        assert out.dtype == DType.bfloat16


def test_moe_fp8_forward(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
            quant_config=fp8_quant_config,
        ).to(device)
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
        )
        out = layer(x)

        assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
        assert out.dtype == DType.bfloat16


def test_moe_forward_with_shared_experts(mock_accelerator: MagicMock) -> None:
    device = mock_accelerator()
    with F.lazy():
        with default_dtype(DType.bfloat16):
            layer = QuantizedMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
                has_shared_experts=True,
                shared_experts_dim=_SHARED_EXPERTS_DIM,
            ).to(device)
            x = Tensor.zeros(
                [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
            )
            out = layer(x)

        assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
        assert out.dtype == DType.bfloat16


# --------------------------------------------------------------------------- #
# NVFP4
# --------------------------------------------------------------------------- #


def test_moe_nvfp4_parameters(
    mock_accelerator: MagicMock, nvfp4_quant_config: QuantConfig
) -> None:
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
            quant_config=nvfp4_quant_config,
        ).to(device)

        names = {name for name, _ in layer.parameters}
        # The gate is never quantized.
        expected = {"gate.gate_score.weight"} | _expected_expert_param_names(
            quantized=True, leaves=_NVFP4_LEAVES
        )
        assert names == expected


def test_moe_stacked_weights_nvfp4(
    mock_accelerator: MagicMock, nvfp4_quant_config: QuantConfig
) -> None:
    """NVFP4 stacking keeps one global scale pair per expert."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
            quant_config=nvfp4_quant_config,
        ).to(device)

        gate_up = layer.gate_up_proj
        assert len(gate_up) == 1
        assert isinstance(gate_up[0], NVFP4Tensor)
        assert gate_up[0].data.dtype == DType.uint8
        assert gate_up[0].weight_scale.dtype == DType.float8_e4m3fn
        assert list(gate_up[0].data.shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM,
            _HIDDEN_DIM // 2,
        ]
        assert list(gate_up[0].weight_scale.shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM,
            _HIDDEN_DIM // 16,
        ]
        # gate and up share one global scale, so stacking gives one per expert.
        assert list(gate_up[0].weight_scale_2.shape) == [_NUM_EXPERTS]
        assert list(gate_up[0].input_scale.shape) == [_NUM_EXPERTS]

        down = layer.down_proj
        assert len(down) == 1
        assert isinstance(down[0], NVFP4Tensor)
        assert list(down[0].data.shape) == [
            _NUM_EXPERTS,
            _HIDDEN_DIM,
            _MOE_DIM // 2,
        ]
        assert list(down[0].weight_scale.shape) == [
            _NUM_EXPERTS,
            _HIDDEN_DIM,
            _MOE_DIM // 16,
        ]


def test_moe_nvfp4_forward(
    mock_accelerator: MagicMock,
    nvfp4_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    """The NVFP4 routed-expert path returns a bf16 ``[S, hidden]`` activation.

    Exercises the NVFP4-only index output (``needs_scales_offset``) feeding the
    grouped FP4 matmuls and the SiLU re-quantization between them.
    """
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
            quant_config=nvfp4_quant_config,
        ).to(device)
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
        )
        out = layer(x)

        assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
        assert out.dtype == DType.bfloat16


def test_moe_nvfp4_forward_with_shared_experts(
    mock_accelerator: MagicMock,
    nvfp4_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    """The quantized shared expert adds into the routed-expert output."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
            has_shared_experts=True,
            shared_experts_dim=_SHARED_EXPERTS_DIM,
            quant_config=nvfp4_quant_config,
        ).to(device)
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
        )
        out = layer(x)

        assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
        assert out.dtype == DType.bfloat16
