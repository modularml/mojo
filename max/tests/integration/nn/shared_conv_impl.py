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

# This code is shared between test_conv.py and test_conv_gpu.py


import numpy as np
import torch
from max.driver import Buffer
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import Conv1D, Conv2d, Conv3D
from torch import nn

# On some newer CUDA architectures (e.g. Ampere / Hopper) cuDNN may internally
ACCURACY_RTOL = 2.5e-3
ACCURACY_ATOL = 1e-5


def conv3d_impl(session: InferenceSession) -> None:
    torch.manual_seed(42)

    in_channels = 3
    out_channels = 1280
    kernel_size = (2, 14, 14)
    stride = (2, 14, 14)

    # input params
    batch_size = 3
    depth = 32
    height = 112
    width = 112

    is_gpu = not session.devices[0].is_host
    torch_dtype = torch.float32
    torch_device = torch.device("cuda") if is_gpu else torch.device("cpu")
    max_dtype = DType.float32
    max_device = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    input_sequence = torch.rand(
        size=(batch_size, in_channels, depth, height, width),
        dtype=torch_dtype,
        device=torch_device,
    )

    torch_conv = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=False,
        device=torch_device,
    )

    max_conv = Conv3D(
        depth=kernel_size[0],
        height=kernel_size[1],
        width=kernel_size[2],
        in_channels=in_channels,
        out_channels=out_channels,
        dtype=max_dtype,
        stride=stride,
        has_bias=False,
        permute=True,
        device=max_device,
    )

    # load random weights to torch
    torch_conv.weight.data = nn.Parameter(
        torch.rand(size=torch_conv.weight.data.shape, device=torch_device)
    )

    # load weights to max
    state_dict = {"weight": torch_conv.weight.data.detach().cpu()}
    max_conv.load_state_dict(state_dict)

    # get_torch_output
    with torch.no_grad():
        torch_conv_result = torch_conv(input_sequence)

    # get_max_output
    graph = Graph(
        "conv3d",
        max_conv,
        input_types=(
            TensorType(max_dtype, input_sequence.shape, device=max_device),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())

    graph_api_conv_result = compiled.execute(input_sequence)[0]
    assert isinstance(graph_api_conv_result, Buffer)

    np.testing.assert_allclose(
        graph_api_conv_result.to_numpy(),
        torch_conv_result.detach().cpu().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


def conv1d_impl(session: InferenceSession) -> None:
    torch.manual_seed(42)

    batch_size = 1
    in_channels = 1024
    length = 57
    hidden_size = 1024  # out_channels
    kernel_size = 7
    stride = 1
    padding = 3

    is_gpu = not session.devices[0].is_host
    torch_dtype = torch.float32
    torch_device = torch.device("cuda") if is_gpu else torch.device("cpu")
    max_dtype = DType.float32
    max_device = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    # Create input tensor in PyTorch format (batch_size, in_channels, length)
    input_sequence = torch.rand(
        size=(batch_size, in_channels, length),
        dtype=torch_dtype,
        device=torch_device,
    )

    # Create PyTorch Conv1d layer
    torch_conv = nn.Conv1d(
        in_channels=in_channels,
        out_channels=hidden_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        device=torch_device,
    )

    # Create our Conv1D layer
    max_conv = Conv1D(
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=hidden_size,
        dtype=max_dtype,
        stride=stride,
        device=max_device,
        padding=padding,
        has_bias=True,
        permute=True,
    )

    # Initialize random weights for PyTorch conv
    torch_conv.weight.data = nn.Parameter(
        torch.rand(
            size=torch_conv.weight.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )
    assert torch_conv.bias is not None

    torch_conv.bias.data = nn.Parameter(
        torch.rand(
            size=torch_conv.bias.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )
    # Load the same weights into our conv
    state_dict = {
        "weight": torch_conv.weight.data.detach().cpu(),
        "bias": torch_conv.bias.data.detach().cpu(),
    }
    max_conv.load_state_dict(state_dict)

    # Get PyTorch output
    with torch.no_grad():
        torch_conv_result = torch_conv(input_sequence)

    # Get Max output
    graph = Graph(
        "conv1d",
        max_conv,
        input_types=(
            TensorType(max_dtype, input_sequence.shape, device=max_device),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())
    graph_api_conv_result = compiled.execute(input_sequence)[0]
    assert isinstance(graph_api_conv_result, Buffer)

    np.testing.assert_allclose(
        graph_api_conv_result.to_numpy(),
        torch_conv_result.detach().cpu().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


def conv1d_tuple_padding_impl(session: InferenceSession) -> None:
    """Test Conv1D with tuple padding (asymmetric padding)."""
    torch.manual_seed(42)

    batch_size = 1
    in_channels = 64
    length = 20
    hidden_size = 128  # out_channels
    kernel_size = 3
    stride = 1
    padding_tuple = (2, 0)  # left padding only

    is_gpu = not session.devices[0].is_host
    torch_dtype = torch.float32
    torch_device = torch.device("cuda") if is_gpu else torch.device("cpu")
    max_dtype = DType.float32
    max_device = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    input_sequence = torch.rand(
        size=(batch_size, in_channels, length),
        dtype=torch_dtype,
        device=torch_device,
    )

    torch_conv = nn.Conv1d(
        in_channels=in_channels,
        out_channels=hidden_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        bias=True,
        device=torch_device,
    )

    max_conv = Conv1D(
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=hidden_size,
        dtype=max_dtype,
        stride=stride,
        device=max_device,
        padding=padding_tuple,
        has_bias=True,
        permute=True,
    )

    torch_conv.weight.data = nn.Parameter(
        torch.rand(
            size=torch_conv.weight.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )
    assert torch_conv.bias is not None
    torch_conv.bias.data = nn.Parameter(
        torch.rand(
            size=torch_conv.bias.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )

    state_dict = {
        "weight": torch_conv.weight.data.detach().cpu(),
        "bias": torch_conv.bias.data.detach().cpu(),
    }
    max_conv.load_state_dict(state_dict)

    with torch.no_grad():
        padded_input = nn.functional.pad(
            input_sequence,
            (padding_tuple[0], padding_tuple[1]),
            mode="constant",
            value=0,
        )
        torch_conv_result = torch_conv(padded_input)

    graph = Graph(
        "conv1d_tuple_padding",
        max_conv,
        input_types=(
            TensorType(max_dtype, input_sequence.shape, device=max_device),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())
    graph_api_conv_result = compiled.execute(input_sequence)[0]
    assert isinstance(graph_api_conv_result, Buffer)

    np.testing.assert_allclose(
        graph_api_conv_result.to_numpy(),
        torch_conv_result.detach().cpu().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


def conv1d_tuple_padding_nonfcrs_impl(session: InferenceSession) -> None:
    """Test Conv1D with tuple padding on non-FCRS (permute=False) path."""
    torch.manual_seed(123)

    batch_size = 2
    in_channels = 8
    length = 16
    hidden_size = 4  # out_channels
    kernel_size = 3
    stride = 1
    padding_tuple = (2, 0)  # left padding only

    is_gpu = not session.devices[0].is_host
    torch_dtype = torch.float32
    torch_device = torch.device("cuda") if is_gpu else torch.device("cpu")
    max_dtype = DType.float32
    max_device = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    # Torch expects (N, C, L); Conv1D permute=False expects (N, L, C).
    torch_input = torch.rand(
        size=(batch_size, in_channels, length),
        dtype=torch_dtype,
        device=torch_device,
    )
    max_input = torch_input.permute(0, 2, 1).contiguous()

    torch_conv = nn.Conv1d(
        in_channels=in_channels,
        out_channels=hidden_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        bias=True,
        device=torch_device,
    )

    max_conv = Conv1D(
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=hidden_size,
        dtype=max_dtype,
        stride=stride,
        device=max_device,
        padding=padding_tuple,
        has_bias=True,
        permute=False,  # non-FCRS layout path
    )

    torch_conv.weight.data = nn.Parameter(
        torch.rand(
            size=torch_conv.weight.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )
    assert torch_conv.bias is not None
    torch_conv.bias.data = nn.Parameter(
        torch.rand(
            size=torch_conv.bias.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )

    # Convert torch weights to RSCF (K, C_in, C_out) layout expected by permute=False.
    state_dict = {
        "weight": torch_conv.weight.data.detach()
        .cpu()
        .permute(2, 1, 0)
        .contiguous(),
        "bias": torch_conv.bias.data.detach().cpu(),
    }
    max_conv.load_state_dict(state_dict)

    with torch.no_grad():
        padded_input = nn.functional.pad(
            torch_input,
            (padding_tuple[0], padding_tuple[1]),
            mode="constant",
            value=0,
        )
        torch_conv_result = torch_conv(padded_input).permute(0, 2, 1)

    graph = Graph(
        "conv1d_tuple_padding_nonfcrs",
        max_conv,
        input_types=(
            TensorType(max_dtype, max_input.shape, device=max_device),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())
    graph_api_conv_result = compiled.execute(max_input)[0]
    assert isinstance(graph_api_conv_result, Buffer)

    np.testing.assert_allclose(
        graph_api_conv_result.to_numpy(),
        torch_conv_result.detach().cpu().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


def conv2d_tuple_padding_impl(session: InferenceSession) -> None:
    """Test Conv2d with tuple padding (asymmetric padding)."""
    torch.manual_seed(7)

    batch_size = 2
    in_channels = 16
    out_channels = 8
    kernel_size = (3, 3)
    stride = (1, 1)
    padding_tuple = (1, 0, 2, 1)  # top, bottom, left, right
    height = 8
    width = 7

    is_gpu = not session.devices[0].is_host
    torch_dtype = torch.float32
    torch_device = torch.device("cuda") if is_gpu else torch.device("cpu")
    max_dtype = DType.float32
    max_device = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    input_sequence = torch.rand(
        size=(batch_size, in_channels, height, width),
        dtype=torch_dtype,
        device=torch_device,
    )

    torch_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        bias=True,
        device=torch_device,
    )

    max_conv = Conv2d(
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        dtype=max_dtype,
        stride=stride,
        padding=padding_tuple,
        has_bias=True,
        permute=True,
        device=max_device,
    )

    torch_conv.weight.data = nn.Parameter(
        torch.rand(
            size=torch_conv.weight.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )
    assert torch_conv.bias is not None
    torch_conv.bias.data = nn.Parameter(
        torch.rand(
            size=torch_conv.bias.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )

    state_dict = {
        "weight": torch_conv.weight.data.detach().cpu(),
        "bias": torch_conv.bias.data.detach().cpu(),
    }
    max_conv.load_state_dict(state_dict)

    with torch.no_grad():
        padded_input = nn.functional.pad(
            input_sequence,
            (
                padding_tuple[2],
                padding_tuple[3],
                padding_tuple[0],
                padding_tuple[1],
            ),
            mode="constant",
            value=0,
        )
        torch_conv_result = torch_conv(padded_input)

    graph = Graph(
        "conv2d_tuple_padding",
        max_conv,
        input_types=(
            TensorType(max_dtype, input_sequence.shape, device=max_device),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())
    graph_api_conv_result = compiled.execute(input_sequence)[0]
    assert isinstance(graph_api_conv_result, Buffer)

    np.testing.assert_allclose(
        graph_api_conv_result.to_numpy(),
        torch_conv_result.detach().cpu().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


def conv3d_tuple_padding_impl(session: InferenceSession) -> None:
    """Test Conv3D with tuple padding (asymmetric padding)."""
    torch.manual_seed(9)

    batch_size = 1
    in_channels = 4
    out_channels = 6
    depth, height, width = 6, 5, 4
    kernel_size = (3, 3, 2)
    stride = (1, 1, 1)
    padding_tuple = (1, 0, 2, 1, 3, 0)  # front, back, top, bottom, left, right

    is_gpu = not session.devices[0].is_host
    torch_dtype = torch.float32
    torch_device = torch.device("cuda") if is_gpu else torch.device("cpu")
    max_dtype = DType.float32
    max_device = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    input_sequence = torch.rand(
        size=(batch_size, in_channels, depth, height, width),
        dtype=torch_dtype,
        device=torch_device,
    )

    torch_conv = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        bias=True,
        device=torch_device,
    )

    max_conv = Conv3D(
        depth=kernel_size[0],
        height=kernel_size[1],
        width=kernel_size[2],
        in_channels=in_channels,
        out_channels=out_channels,
        dtype=max_dtype,
        stride=stride,
        padding=padding_tuple,
        has_bias=True,
        permute=True,
        device=max_device,
    )

    torch_conv.weight.data = nn.Parameter(
        torch.rand(
            size=torch_conv.weight.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )
    assert torch_conv.bias is not None
    torch_conv.bias.data = nn.Parameter(
        torch.rand(
            size=torch_conv.bias.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )

    state_dict = {
        "weight": torch_conv.weight.data.detach().cpu(),
        "bias": torch_conv.bias.data.detach().cpu(),
    }
    max_conv.load_state_dict(state_dict)

    with torch.no_grad():
        padded_input = nn.functional.pad(
            input_sequence,
            (
                padding_tuple[4],
                padding_tuple[5],
                padding_tuple[2],
                padding_tuple[3],
                padding_tuple[0],
                padding_tuple[1],
            ),
            mode="constant",
            value=0,
        )
        torch_conv_result = torch_conv(padded_input)

    graph = Graph(
        "conv3d_tuple_padding",
        max_conv,
        input_types=(
            TensorType(max_dtype, input_sequence.shape, device=max_device),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())
    graph_api_conv_result = compiled.execute(input_sequence)[0]
    assert isinstance(graph_api_conv_result, Buffer)

    np.testing.assert_allclose(
        graph_api_conv_result.to_numpy(),
        torch_conv_result.detach().cpu().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


def conv2d_1x1_impl(session: InferenceSession) -> None:
    """Test 1x1 Conv2d with permute=True (FCRS layout).

    This specifically exercises the conv-to-matmul optimization in the graph
    compiler, which rewrites 1x1 convolutions as matmul ops. A past bug
    caused the FCRS filter's transpose to be dropped by
    MatmulOp::canonicalize, producing silently wrong results.
    """
    torch.manual_seed(99)

    batch_size = 2
    in_channels = 16
    out_channels = 32
    kernel_size = (1, 1)
    stride = (1, 1)
    padding = 0
    height = 8
    width = 8

    is_gpu = not session.devices[0].is_host
    torch_dtype = torch.float32
    torch_device = torch.device("cuda") if is_gpu else torch.device("cpu")
    max_dtype = DType.float32
    max_device = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    input_sequence = torch.rand(
        size=(batch_size, in_channels, height, width),
        dtype=torch_dtype,
        device=torch_device,
    )

    torch_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        device=torch_device,
    )

    max_conv = Conv2d(
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        dtype=max_dtype,
        stride=stride,
        padding=padding,
        has_bias=True,
        permute=True,
        device=max_device,
    )

    torch_conv.weight.data = nn.Parameter(
        torch.rand(
            size=torch_conv.weight.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )
    assert torch_conv.bias is not None
    torch_conv.bias.data = nn.Parameter(
        torch.rand(
            size=torch_conv.bias.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )

    state_dict = {
        "weight": torch_conv.weight.data.detach().cpu(),
        "bias": torch_conv.bias.data.detach().cpu(),
    }
    max_conv.load_state_dict(state_dict)

    with torch.no_grad():
        torch_conv_result = torch_conv(input_sequence)

    graph = Graph(
        "conv2d_1x1",
        max_conv,
        input_types=(
            TensorType(max_dtype, input_sequence.shape, device=max_device),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())
    graph_api_conv_result = compiled.execute(input_sequence)[0]
    assert isinstance(graph_api_conv_result, Buffer)

    np.testing.assert_allclose(
        graph_api_conv_result.to_numpy(),
        torch_conv_result.detach().cpu().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


def conv2d_impl(session: InferenceSession) -> None:
    torch.manual_seed(42)

    batch_size = 4
    in_channels = 64
    out_channels = 128
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = 1  # Will be converted to (1, 1, 1, 1)
    height = 32
    width = 32

    is_gpu = not session.devices[0].is_host
    torch_dtype = torch.float32
    torch_device = torch.device("cuda") if is_gpu else torch.device("cpu")
    max_dtype = DType.float32
    max_device = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    # Create input tensor in PyTorch format (batch_size, in_channels, height, width)
    input_sequence = torch.rand(
        size=(batch_size, in_channels, height, width),
        dtype=torch_dtype,
        device=torch_device,
    )

    # Create PyTorch Conv2d layer
    torch_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        device=torch_device,
    )

    # Create our Conv2d layer
    max_conv = Conv2d(
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        dtype=max_dtype,
        stride=stride,
        padding=padding,
        has_bias=True,
        permute=True,
        device=max_device,
    )

    # Initialize random weights for PyTorch conv
    torch_conv.weight.data = nn.Parameter(
        torch.rand(
            size=torch_conv.weight.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )
    assert torch_conv.bias is not None

    torch_conv.bias.data = nn.Parameter(
        torch.rand(
            size=torch_conv.bias.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )

    # Load the same weights into our conv
    state_dict = {
        "weight": torch_conv.weight.data.detach().cpu(),
        "bias": torch_conv.bias.data.detach().cpu(),
    }
    max_conv.load_state_dict(state_dict)

    # Get PyTorch output
    with torch.no_grad():
        torch_conv_result = torch_conv(input_sequence)

    # Get Max output
    graph = Graph(
        "conv2d",
        max_conv,
        input_types=(
            TensorType(max_dtype, input_sequence.shape, device=max_device),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())
    graph_api_conv_result = compiled.execute(input_sequence)[0]
    assert isinstance(graph_api_conv_result, Buffer)

    np.testing.assert_allclose(
        graph_api_conv_result.to_numpy(),
        torch_conv_result.detach().cpu().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )
