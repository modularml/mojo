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
"""Test MLP layer with FP8 quantization using fbgemm_fp8 config.

Tests the original MLP implementation path (lines 830-833 in linear.py):
    return self.down_proj(
        self.activation_function(self.gate_proj(TensorValue(x)))
        * self.up_proj(TensorValue(x))
    )

Uses float8 scaling/config matching RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic:
- Input scaling: Dynamic, column-wise (per-token)
- Weight scaling: Static, row-wise
- Quantization method: fbgemm_fp8
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType, TensorValue
from max.graph.weights import WeightData
from max.nn import MLP
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from torch import nn


def _skip_if_no_gpu() -> None:
    """Skip test if no GPU is available."""
    if accelerator_count() == 0:
        pytest.skip("No GPU available for FP8 MLP test")


def _create_fbgemm_fp8_config() -> QuantConfig:
    """Creates QuantConfig.
    - Dynamic, column-wise (per-token) input scaling
    - Static, row-wise weight scaling
    """
    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.COLWISE,
        origin=ScaleOrigin.DYNAMIC,
        dtype=DType.float32,
        activation_scale_ub=None,
    )
    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.ROWWISE,
        dtype=DType.float32,
    )
    return QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_quantized_layers={0},
        attn_quantized_layers=set(),
        embedding_output_dtype=DType.bfloat16,
        format=QuantFormat.FBGEMM_FP8,
    )


def _generate_weights(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    seed: int,
    scale: float = 0.1,
) -> torch.Tensor:
    """Generates random weights with reproducible seeding.

    Args:
        shape: Shape of the weight tensor.
        dtype: Data type of the weight tensor.
        seed: Random seed for reproducibility.
        scale: Scale factor for the random values.

    Returns:
        Random tensor with the specified shape and dtype.
    """
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=dtype) * scale


def _quantize_to_fp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantizes weights to FP8 with row-wise scaling.

    Args:
        weight: BFloat16 or Float32 weight tensor of shape [out_features, in_features].

    Returns:
        Tuple of (fp8_weight, weight_scale) where weight_scale is [out_features, 1].
    """
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    weight_fp32 = weight.float()
    abs_max = weight_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = fp8_max / abs_max
    weight_scale = (1.0 / scale).to(torch.float32)
    quantized = (weight_fp32 * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)

    return quantized, weight_scale


class TorchMLPFloat8Reference(nn.Module):
    """PyTorch reference MLP using the original (non-fused) implementation.

    This matches the implementation in linear.py lines 830-833:
        return self.down_proj(
            self.activation_function(self.gate_proj(TensorValue(x)))
            * self.up_proj(TensorValue(x))
        )
    """

    def __init__(
        self,
        gate_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        gate_proj_scale: torch.Tensor,
        down_proj_scale: torch.Tensor,
        up_proj_scale: torch.Tensor,
        activation_function: str = "silu",
    ) -> None:
        """Initializes the reference MLP.

        Args:
            gate_proj_weight: FP8 gate projection weights [ffl, hidden].
            down_proj_weight: FP8 down projection weights [hidden, ffl].
            up_proj_weight: FP8 up projection weights [ffl, hidden].
            gate_proj_scale: Row-wise scale for gate projection.
            down_proj_scale: Row-wise scale for down projection.
            up_proj_scale: Row-wise scale for up projection.
            activation_function: Activation function name.
        """
        super().__init__()
        self.gate_proj_weight = gate_proj_weight
        self.down_proj_weight = down_proj_weight
        self.up_proj_weight = up_proj_weight
        self.gate_proj_scale = gate_proj_scale
        self.down_proj_scale = down_proj_scale
        self.up_proj_scale = up_proj_scale
        self.activation_function = activation_function

    def _dequantize(
        self, weight: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Dequantizes FP8 weights back to bfloat16."""
        return (weight.float() * scale).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the MLP transformation.

        Args:
            x: Input tensor of shape [batch, hidden_dim].

        Returns:
            Output tensor of shape [batch, hidden_dim].
        """
        gate_w = self._dequantize(self.gate_proj_weight, self.gate_proj_scale)
        up_w = self._dequantize(self.up_proj_weight, self.up_proj_scale)
        down_w = self._dequantize(self.down_proj_weight, self.down_proj_scale)

        gate_out = F.linear(x, gate_w)
        if self.activation_function == "silu":
            gate_out = F.silu(gate_out)
        elif self.activation_function == "gelu":
            gate_out = F.gelu(gate_out)
        else:
            raise ValueError(
                f"Unsupported activation function: {self.activation_function}"
            )
        up_out = F.linear(x, up_w)
        hidden = gate_out * up_out
        return F.linear(hidden, down_w)


def _wrap_fp8_tensor(tensor: torch.Tensor, name: str) -> WeightData:
    """Wraps an FP8 tensor as WeightData for dlpack compatibility.

    FP8 tensors cannot be directly converted via dlpack, so we view them
    as uint8 first, then convert back to FP8 dtype.

    Args:
        tensor: FP8 tensor to wrap.
        name: Name for the weight.

    Returns:
        WeightData wrapping the FP8 tensor.
    """
    uint8_tensor = tensor.cpu().view(torch.uint8)
    return WeightData(
        Buffer.from_dlpack(uint8_tensor).view(DType.float8_e4m3fn),
        name,
        DType.float8_e4m3fn,
        Shape(tensor.shape),
    )


def _create_mlp_state_dict(
    gate_proj_fp8: torch.Tensor,
    gate_proj_scale: torch.Tensor,
    down_proj_fp8: torch.Tensor,
    down_proj_scale: torch.Tensor,
    up_proj_fp8: torch.Tensor,
    up_proj_scale: torch.Tensor,
) -> dict[str, WeightData | torch.Tensor]:
    """Creates state dict for MAX MLP layer with FP8 weights.

    Args:
        gate_proj_fp8: FP8 gate projection weights.
        gate_proj_scale: Scale for gate projection.
        down_proj_fp8: FP8 down projection weights.
        down_proj_scale: Scale for down projection.
        up_proj_fp8: FP8 up projection weights.
        up_proj_scale: Scale for up projection.

    Returns:
        State dict mapping weight names to WeightData or tensors.
    """
    return {
        "gate_proj.weight": _wrap_fp8_tensor(gate_proj_fp8, "gate_proj.weight"),
        "gate_proj.weight_scale": gate_proj_scale.cpu(),
        "down_proj.weight": _wrap_fp8_tensor(down_proj_fp8, "down_proj.weight"),
        "down_proj.weight_scale": down_proj_scale.cpu(),
        "up_proj.weight": _wrap_fp8_tensor(up_proj_fp8, "up_proj.weight"),
        "up_proj.weight_scale": up_proj_scale.cpu(),
    }


def test_mlp_float8_fbgemm() -> None:
    """Tests MLP layer with fbgemm_fp8 quantization config."""
    _skip_if_no_gpu()
    hidden_dim = 256
    feed_forward_length = 512
    activation = "silu"

    device = Accelerator(0)
    device_ref = DeviceRef(device.label, device.id)
    quant_config = _create_fbgemm_fp8_config()

    gate_proj_bf16 = _generate_weights(
        (feed_forward_length, hidden_dim), torch.bfloat16, seed=42
    )
    down_proj_bf16 = _generate_weights(
        (hidden_dim, feed_forward_length), torch.bfloat16, seed=43
    )
    up_proj_bf16 = _generate_weights(
        (feed_forward_length, hidden_dim), torch.bfloat16, seed=44
    )

    gate_proj_fp8, gate_scale = _quantize_to_fp8(gate_proj_bf16)
    down_proj_fp8, down_scale = _quantize_to_fp8(down_proj_bf16)
    up_proj_fp8, up_scale = _quantize_to_fp8(up_proj_bf16)

    state_dict = _create_mlp_state_dict(
        gate_proj_fp8,
        gate_scale,
        down_proj_fp8,
        down_scale,
        up_proj_fp8,
        up_scale,
    )

    mlp = MLP(
        dtype=DType.float8_e4m3fn,
        quantization_encoding=None,
        hidden_dim=hidden_dim,
        feed_forward_length=feed_forward_length,
        devices=[device_ref],
        activation_function=activation,
        has_bias=False,
        quant_config=quant_config,
    )
    mlp.load_state_dict(state_dict)

    seq_len = 4
    x_scale = 1.0 / math.sqrt(hidden_dim)
    x = (
        _generate_weights((seq_len, hidden_dim), torch.bfloat16, seed=45)
        * x_scale
    )
    x = x.to(torch.bfloat16).cuda()

    session = InferenceSession(devices=[device])
    with Graph(
        "MLP_FP8_Test",
        input_types=[
            TensorType(
                DType.bfloat16,
                (seq_len, hidden_dim),
                device=device_ref,
            ),
        ],
    ) as graph:
        (graph_input,) = graph.inputs
        assert isinstance(graph_input, TensorValue)
        graph_output = mlp(graph_input)
        graph.output(graph_output)

    torch_ref = TorchMLPFloat8Reference(
        gate_proj_fp8.cuda(),
        down_proj_fp8.cuda(),
        up_proj_fp8.cuda(),
        gate_scale.cuda(),
        down_scale.cuda(),
        up_scale.cuda(),
        activation_function=activation,
    )
    torch_output = torch_ref(x).cpu()
    compiled = session.load(graph, weights_registry=mlp.state_dict())
    max_output = compiled.execute(x)[0]

    assert isinstance(max_output, Buffer)
    max_result = torch.from_dlpack(max_output).float().cpu()
    torch_result = torch_output.float()

    assert torch.isfinite(max_result).all(), "MAX output contains NaN or Inf"
    torch.testing.assert_close(
        max_result,
        torch_result,
        rtol=1e-4,
        atol=1 * torch.finfo(torch.bfloat16).eps,
    )
