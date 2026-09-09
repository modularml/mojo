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
import numpy as np
import pytest
import torch
from max.driver import Buffer
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import Linear
from torch import nn

ACCURACY_RTOL = 1e-3
ACCURACY_ATOL = 2e-3


# This corresponds to M, N, K, transpose_b=True for the underlying matmul
@pytest.mark.parametrize(
    "sequence_length, out_features, in_features",
    [
        # (M, N, K) -> (sequence_length, out_features, in_features)
        (64, 1024, 2048),
    ],
)
def test_linear_gpu(
    gpu_session: InferenceSession,
    sequence_length: int,
    out_features: int,
    in_features: int,
) -> None:
    linear_impl(gpu_session, out_features, in_features, sequence_length)


def linear_impl(
    session: InferenceSession,
    out_features: int,
    in_features: int,
    sequence_length: int,
) -> None:
    torch.manual_seed(42)
    batch_size = 1
    # Multiple models use std=0.02 for initialization, including GPT-2 and BERT
    std = 0.02

    has_bias = True

    is_gpu = not session.devices[0].is_host
    torch_dtype = torch.float32
    torch_device = torch.device("cuda") if is_gpu else torch.device("cpu")
    max_dtype = DType.float32
    max_device = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    # Distribution matches post-LayerNorm distribution (mean=0, std=1)
    input_tensor = torch.randn(
        size=(batch_size, sequence_length, in_features),
        dtype=torch_dtype,
        device=torch_device,
    )

    torch_linear = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=has_bias,
        device=torch_device,
    )

    max_linear = Linear(
        in_dim=in_features,
        out_dim=out_features,
        dtype=max_dtype,
        has_bias=has_bias,
        device=max_device,
    )

    torch_linear.weight.data = std * torch.randn_like(torch_linear.weight)

    if has_bias:
        assert torch_linear.bias is not None
        torch_linear.bias.data = std * torch.randn_like(torch_linear.bias)

    # Load the same weights into MAX linear
    state_dict = {
        "weight": torch_linear.weight.data.detach().cpu(),
    }
    if has_bias:
        state_dict["bias"] = torch_linear.bias.data.detach().cpu()

    max_linear.load_state_dict(state_dict)

    # Get PyTorch output
    with torch.no_grad():
        torch_linear_result = torch_linear(input_tensor)

    # Get MAX output
    graph = Graph(
        "linear",
        max_linear,
        input_types=(
            TensorType(max_dtype, input_tensor.shape, device=max_device),
        ),
    )

    compiled = session.load(graph, weights_registry=max_linear.state_dict())
    graph_api_linear_result = compiled.execute(input_tensor)[0]
    assert isinstance(graph_api_linear_result, Buffer)

    np.testing.assert_allclose(
        graph_api_linear_result.to_numpy(),
        torch_linear_result.detach().cpu().numpy(),
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )
