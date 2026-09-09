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
from max import nn
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType

device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()


@pytest.mark.parametrize("dtype", [DType.float32, DType.int16])
def test_clamp(session: InferenceSession, dtype: DType) -> None:
    """Test clamp with all parameter combinations: both, min only, max only, none"""
    input_type = TensorType(dtype, [10, 10], device=device_ref)

    with Graph(f"clamp_{dtype}", input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor

        both_bounds = nn.clamp(x, min=10, max=20).cast(DType.float32)
        no_bounds = nn.clamp(x).cast(DType.float32)
        min_only = nn.clamp(x, min=10).cast(DType.float32)
        max_only = nn.clamp(x, max=50).cast(DType.float32)

        graph.output(both_bounds, no_bounds, min_only, max_only)

    model = session.load(graph)

    torch_dtype = torch.float32 if dtype == DType.float32 else torch.int16
    input_data = torch.arange(end=100, dtype=torch_dtype).reshape((10, 10))

    results = model(Buffer.from_dlpack(input_data).to(model.input_devices[0]))

    # Expected results from PyTorch
    both_bounds_expected = (
        torch.clamp(input_data, min=10, max=20)
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )
    no_bounds_expected = input_data.to(dtype=torch.float32).cpu().numpy()
    min_only_expected = (
        torch.clamp(input_data, min=10).to(dtype=torch.float32).cpu().numpy()
    )
    max_only_expected = (
        torch.clamp(input_data, max=50).to(dtype=torch.float32).cpu().numpy()
    )

    assert len(results) == 4
    tensor_results = []
    for i, result in enumerate(results):
        assert isinstance(result, Buffer), f"Result {i} is not a Tensor"
        tensor_results.append(result)

    np.testing.assert_allclose(
        tensor_results[0].to_numpy(), both_bounds_expected, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        tensor_results[1].to_numpy(), no_bounds_expected, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        tensor_results[2].to_numpy(), min_only_expected, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        tensor_results[3].to_numpy(), max_only_expected, rtol=1e-6, atol=1e-6
    )
