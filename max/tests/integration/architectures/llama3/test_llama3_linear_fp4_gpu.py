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

"""Test for FP4 linear layers with NVFP4 quantized checkpoint loading."""

import json
import os
from pathlib import Path

import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType
from max.graph.weights import WeightData
from max.nn import Linear
from max.pipelines.weights.quant import parse_quant_config
from test_common.graph_utils import is_h100_h200
from torch.utils.dlpack import from_dlpack
from transformers.models.llama.configuration_llama import LlamaConfig

RTOL = 0.01
ATOL = 0.01


@pytest.fixture
def config() -> LlamaConfig:
    config = LlamaConfig()
    path = os.environ["PIPELINES_TESTDATA"]
    config_path = Path(path) / "config_nvfp4.json"
    with open(config_path) as file:
        data = json.load(file)
    config.update(data)
    return config


@pytest.fixture
def model_name() -> str:
    """Model name for FP4 Llama3 model."""
    return "nvidia/Llama-3.1-8B-Instruct-NVFP4"


@pytest.fixture
def device() -> Accelerator:
    """Device to run the model on."""
    return Accelerator()


@pytest.fixture
def input_tensor(
    batch_size: int = 1,
    seq_len: int = 10,
    hidden_size: int = 4096,
    seed: int = 42,
) -> torch.Tensor:
    """Generate dummy input tensor."""
    torch.manual_seed(seed)
    return torch.randn(
        batch_size,
        seq_len,
        hidden_size,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def input_scale() -> torch.Tensor:
    return torch.tensor([0.0015], dtype=torch.float32)


@pytest.fixture
def weight_scale_2() -> torch.Tensor:
    return torch.tensor([0.0002], dtype=torch.float32)


@pytest.fixture
def weight(
    outdim: int = 14336,
    indim: int = 4096,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(
        255,
        (outdim, indim // 2),
        dtype=torch.uint8,
    )


@pytest.fixture
def weight_scale(
    outdim: int = 14336,
    indim: int = 4096,
    block_size: int = 16,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return (
        torch.randn(
            outdim,
            indim // block_size,
            dtype=torch.bfloat16,
        )
        .abs()
        .to(torch.float8_e4m3fn)
    )


def get_state_dict(
    layer_name: str,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> dict[str, WeightData]:
    return {
        f"{layer_name}.input_scale": WeightData(
            input_scale,
            f"{layer_name}.input_scale",
            DType.float32,
            Shape(input_scale.shape),
        ),
        f"{layer_name}.weight": WeightData(
            weight, f"{layer_name}.weight", DType.uint8, Shape(weight.shape)
        ),
        f"{layer_name}.weight_scale": WeightData(
            Buffer.from_dlpack(weight_scale.view(torch.uint8)).view(
                DType.float8_e4m3fn
            ),  # avoids BufferError: float8 types are not supported by dlpack, when loading state_dict
            f"{layer_name}.weight_scale",
            DType.float8_e4m3fn,
            Shape(weight_scale.shape),
        ),
        f"{layer_name}.weight_scale_2": WeightData(
            weight_scale_2,
            f"{layer_name}.weight_scale_2",
            DType.float32,
            Shape(weight_scale_2.shape),
        ),
    }


def generate_max_linear_output(
    state_dict: dict[str, WeightData],
    config: LlamaConfig,
    input_tensor: torch.Tensor,
    layer_name: str,
    device: Accelerator,
) -> torch.Tensor:
    """Generate output using MAX Linear layer with FP4 quantization.

    Args:
        state_dict: State dictionary with weight and weight_scale
        config: Model config
        input_tensor: Input tensor
        layer_name: Name of the linear layer
        device: Device to run on

    Returns:
        Output tensor from the linear layer
    """
    # Parse quant config for fp4
    quant_config = parse_quant_config(
        config,
        state_dict,
        DType.uint8,  # uint8 for fp4-e2m1fnX2
    )

    if quant_config is None:
        raise ValueError("Failed to parse quant config for FP4")

    # Get weight shape to determine in_dim and out_dim
    weight_key = f"{layer_name}.weight"
    if weight_key not in state_dict:
        raise ValueError(f"Weight {weight_key} not found in state_dict")

    weight_data = state_dict[weight_key]
    # For fp4, weight shape is [out_dim, in_dim // 2] (uint8 contains 2 fp4 values)
    out_dim, in_dim_half = weight_data.shape
    in_dim = int(in_dim_half) * 2

    # Create Linear layer with fp4 config
    linear = Linear(
        in_dim=in_dim,
        out_dim=int(out_dim),
        dtype=DType.uint8,  # fp4-e2m1fnX2
        device=DeviceRef.GPU(),
        has_bias=False,
        quant_config=quant_config,
        name=layer_name,
    )

    # Load weights
    linear.load_state_dict(state_dict, override_quantization_encoding=True)
    # Build graph
    device_ref = DeviceRef.GPU()
    hidden_size = input_tensor.shape[2]

    input_type = TensorType(
        dtype=DType.bfloat16,
        shape=["total_seq_len", hidden_size],
        device=device_ref,
    )

    session = InferenceSession(devices=[device])

    with Graph("FP4Linear", input_types=(input_type,)) as graph:
        inputs = graph.inputs[0]
        output = linear(inputs.tensor)
        graph.output(output)

    # Compile and execute
    compiled = session.load(graph, weights_registry=linear.state_dict())

    # Flatten batch and sequence dimensions
    batch_size, seq_len, hidden_size = input_tensor.shape
    input_flat = input_tensor.reshape(-1, hidden_size)

    out = compiled.execute(
        Buffer.from_dlpack(input_flat).to(device),
    )[0]

    output_tensor = from_dlpack(out).to(torch.bfloat16)

    # Reshape back
    output = output_tensor.reshape(batch_size, seq_len, -1)

    return output


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="FP4 kernel only supports Nvidia GPUs"
)
@pytest.mark.skipif(
    is_h100_h200(),
    reason="FP4 kernel requires SM100 (B200), not supported on H100/H200 (SM90)",
)
def test_fp4_linear_layer(
    config: LlamaConfig,
    input_tensor: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    device: Accelerator,
    layer_name: str = "model.layers.0.mlp.gate_proj",
) -> None:
    """Test FP4 linear layer by loading NVFP4 weights and scales and running forward pass."""
    state_dict = get_state_dict(
        layer_name, input_scale, weight, weight_scale, weight_scale_2
    )

    # Generate MAX output
    max_output = generate_max_linear_output(
        state_dict, config, input_tensor, layer_name, device
    )

    # Check that outputs are not all zeros or NaN
    assert not torch.all(max_output == 0.0), (
        "MAX output should not be all zeros"
    )
    assert not torch.any(torch.isnan(max_output)), (
        "MAX output should not contain NaN"
    )
    assert not torch.any(torch.isinf(max_output)), (
        "MAX output should not contain Inf"
    )
