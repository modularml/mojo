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
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import ConvTranspose1d, WeightNormConvTranspose1d
from test_common.numerics import pytorch_disable_tf32_dtype
from torch import nn

ACCURACY_RTOL = 1e-4
ACCURACY_ATOL = 1e-6


@pytorch_disable_tf32_dtype
def test_conv_transpose1d() -> None:
    batch_size = 1
    in_channels = 16
    length = 3
    out_channels = 32  # out_channels. Has nothing to do with input size
    kernel_size = 5
    stride = 1
    padding = 0
    dilation = 1
    output_padding = 0

    is_gpu = True
    torch_dtype = torch.float32
    torch_device = torch.device("cuda") if is_gpu else torch.device("cpu")
    max_dtype = DType.float32
    max_device = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    # batch_size, in_channels, seq_length = 3000
    input_sequence = torch.rand(
        size=(batch_size, in_channels, length),
        dtype=torch_dtype,
        device=torch_device,
    )

    torch_conv = nn.ConvTranspose1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        bias=False,
        device=torch_device,
    )

    max_conv = ConvTranspose1d(
        length=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        dtype=max_dtype,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        permute=True,
        device=max_device,
    )

    # load random weights to torch
    torch_conv.weight.data = nn.Parameter(
        torch.rand(
            size=torch_conv.weight.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )

    # load weights to max
    state_dict = {"weight": torch_conv.weight.data.detach().cpu()}
    max_conv.load_state_dict(state_dict)

    # get_torch_output
    with torch.no_grad():
        torch_conv_result = torch_conv(input_sequence)

    # get_max_output
    session = InferenceSession(devices=[Accelerator()])
    graph = Graph(
        "conv_transpose1d",
        max_conv,
        input_types=(
            TensorType(max_dtype, input_sequence.shape, device=max_device),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())

    max_conv_result = compiled.execute(input_sequence)[0]
    assert isinstance(max_conv_result, Buffer)

    np.testing.assert_allclose(
        max_conv_result.to_numpy(),
        torch_conv_result.detach().cpu().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


@pytorch_disable_tf32_dtype
def test_conv_transpose1d_bias() -> None:
    batch_size = 10
    in_channels = 16
    length = 3
    out_channels = 33  # out_channels. Has nothing to do with input size
    kernel_size = 5
    stride = 2
    padding = 3
    dilation = 1
    output_padding = 0

    is_gpu = True
    torch_dtype = torch.float32
    torch_device = torch.device("cuda") if is_gpu else torch.device("cpu")
    max_dtype = DType.float32
    max_device = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    # batch_size, in_channels, seq_length = 3000
    input_sequence = torch.rand(
        size=(batch_size, in_channels, length),
        dtype=torch_dtype,
        device=torch_device,
    )

    torch_conv = nn.ConvTranspose1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        bias=True,
    )

    max_conv = ConvTranspose1d(
        length=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        dtype=max_dtype,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        permute=True,
        has_bias=True,
        device=max_device,
    )

    # load random weights to torch
    torch_conv.weight.data = nn.Parameter(
        torch.rand(
            size=torch_conv.weight.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )

    torch_conv.bias.data = nn.Parameter(
        torch.rand(
            size=torch_conv.bias.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )

    # load weights to max
    state_dict = {
        "weight": torch_conv.weight.data.detach().cpu(),
        "bias": torch_conv.bias.data.detach().cpu(),
    }

    max_conv.load_state_dict(state_dict)

    # get_torch_output
    with torch.no_grad():
        torch_conv_result = torch_conv(input_sequence)

    # get_max_output
    session = InferenceSession(devices=[Accelerator()])
    graph = Graph(
        "conv_transpose1d",
        max_conv,
        input_types=(
            TensorType(max_dtype, input_sequence.shape, device=max_device),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())

    max_conv_result = compiled.execute(input_sequence)[0]
    assert isinstance(max_conv_result, Buffer)

    np.testing.assert_allclose(
        max_conv_result.to_numpy(),
        torch_conv_result.detach().cpu().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


@pytorch_disable_tf32_dtype
def test_weight_norm_conv_transpose1d() -> None:
    batch_size = 10
    in_channels = 16
    length = 3
    out_channels = 33  # out_channels. Has nothing to do with input size
    kernel_size = 5
    stride = 2
    padding = 3
    dilation = 1
    output_padding = 0

    is_gpu = True
    torch_dtype = torch.float32
    torch_device = torch.device("cuda") if is_gpu else torch.device("cpu")
    max_dtype = DType.float32
    max_device = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    # Create input tensor
    input_sequence = torch.rand(
        size=(batch_size, in_channels, length),
        dtype=torch_dtype,
        device=torch_device,
    )

    # Create PyTorch model with weight norm
    torch_conv = nn.ConvTranspose1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        bias=True,
    )

    torch_conv = torch.nn.utils.weight_norm(torch_conv)

    # Create MAX model
    max_conv = WeightNormConvTranspose1d(
        length=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        dtype=max_dtype,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        permute=True,
        has_bias=True,
        device=max_device,
    )

    # load random weights to torch
    torch_conv.weight_v.data = nn.Parameter(
        torch.rand(
            size=torch_conv.weight_v.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )
    torch_conv.weight_g.data = nn.Parameter(
        torch.rand(
            size=torch_conv.weight_g.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )
    torch_conv.bias.data = nn.Parameter(
        torch.rand(
            size=torch_conv.bias.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )

    # load weights to max
    state_dict = {
        "weight_v": torch_conv.weight_v.data.detach().cpu(),
        "weight_g": torch_conv.weight_g.data.detach().cpu(),
        "bias": torch_conv.bias.data.detach().cpu(),
    }
    max_conv.load_state_dict(state_dict)

    # Get PyTorch output
    with torch.no_grad():
        torch_conv_result = torch_conv(input_sequence)

    # Get MAX output
    session = InferenceSession(devices=[Accelerator()])
    graph = Graph(
        "weight_norm_conv_transpose1d",
        max_conv,
        input_types=(
            TensorType(max_dtype, input_sequence.shape, device=max_device),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())

    max_conv_result = compiled.execute(input_sequence)[0]
    assert isinstance(max_conv_result, Buffer)

    # Compare outputs
    np.testing.assert_allclose(
        max_conv_result.to_numpy(),
        torch_conv_result.detach().cpu().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )
