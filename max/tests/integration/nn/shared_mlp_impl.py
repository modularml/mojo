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

import math
from collections.abc import Sequence
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn import MLP, Allreduce, Module, Signals
from test_common.graph_utils import are_all_buffer_values_sequence
from torch import nn

DTYPE = DType.float32
TORCH_DTYPE = torch.float32

ACTIVATION_FUNCTION = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
}

ACCURACY_RTOL = 1e-4
ACCURACY_ATOL = 1e-6


def generate_tensor(
    shape: tuple[int, ...], dtype: DType, hidden_dim: int, seed: int = 1234
) -> torch.Tensor:
    torch.manual_seed(seed)  # Set fixed seed for reproducibility
    return torch.randn(shape, dtype=dtype) * (1.0 / math.sqrt(hidden_dim))


def torch_linear(
    weight: torch.Tensor, bias_tensor: torch.Tensor | None = None, **kwargs: Any
) -> nn.Linear:
    linear = nn.Linear(*weight.shape, **kwargs)
    linear.weight = nn.Parameter(weight)
    if bias_tensor is not None:
        linear.bias = nn.Parameter(bias_tensor)
    return linear


class TorchMLP(nn.Module):
    def __init__(
        self,
        gate_proj: torch.Tensor,
        down_proj: torch.Tensor,
        up_proj: torch.Tensor,
        activation_function: str = "silu",
        bias: bool = False,
        bias_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
        swiglu_limit: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate_proj = torch_linear(
            gate_proj,
            bias_tensor=bias_tensors[0] if bias_tensors is not None else None,
            bias=bias,
        )
        self.down_proj = torch_linear(
            down_proj,
            bias_tensor=bias_tensors[1] if bias_tensors is not None else None,
            bias=bias,
        )
        self.up_proj = torch_linear(
            up_proj,
            bias_tensor=bias_tensors[2] if bias_tensors is not None else None,
            bias=bias,
        )
        self.activation_function = activation_function
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = ACTIVATION_FUNCTION[self.activation_function](self.gate_proj(x))
        up = self.up_proj(x)
        if self.swiglu_limit > 0:
            gate = torch.clamp(gate, max=self.swiglu_limit)
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(gate * up)


class WrapModuleForSubgraph(Module):
    def __init__(self, module: Module) -> None:
        super().__init__()
        # The name of the variable is used to determine the prefix of the weights in the Module class
        self.prefix = module

    def __call__(self, *args: Any) -> Value | list[Value] | list[TensorValue]:  # type: ignore[type-arg]
        subgraph = self.prefix.build_subgraph(
            name="subgraph",
            inputs=list(args),
            weight_prefix="prefix.",
        )
        return ops.call(subgraph, *args, prefix="prefix.")


def _distribute_value(
    v: TensorValue, devices: Sequence[Device]
) -> Sequence[TensorValue]:
    return [v.to(DeviceRef(device.label, device.id)) for device in devices]


def mlp_output(
    gate_proj: torch.Tensor,
    down_proj: torch.Tensor,
    up_proj: torch.Tensor,
    x: torch.Tensor,
    activation_function: str,
    dtype: DType,
    n_gpus: int = 0,
    has_bias: bool = False,
    bias: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    use_subgraphs: bool = True,
    swiglu_limit: float = 0.0,
) -> Sequence[Buffer]:
    # Initialize the device-contexts
    devices: list[Device] = (
        [Accelerator(id) for id in range(n_gpus)] if n_gpus > 0 else [CPU(0)]
    )
    devices_refs = [DeviceRef(device.label, device.id) for device in devices]

    state_dict = {
        "gate_proj.weight": gate_proj.cpu(),
        "down_proj.weight": down_proj.cpu(),
        "up_proj.weight": up_proj.cpu(),
    }
    if has_bias:
        assert bias is not None
        gate_proj_b, down_proj_b, up_proj_b = bias
        state_dict.update(
            {
                "gate_proj.bias": gate_proj_b.cpu(),
                "down_proj.bias": down_proj_b.cpu(),
                "up_proj.bias": up_proj_b.cpu(),
            }
        )

    mlp: MLP | WrapModuleForSubgraph

    mlp = MLP(
        dtype,
        None,
        gate_proj.shape[1],
        gate_proj.shape[0],
        devices=devices_refs,
        activation_function=activation_function,
        has_bias=has_bias,
        swiglu_limit=swiglu_limit,
    )

    if n_gpus > 1:
        mlp.sharding_strategy = ShardingStrategy.tensor_parallel(n_gpus)
        mlp_shards = mlp.shard(devices_refs)
        mlp_allreduce = Allreduce(num_accelerators=n_gpus)

    if use_subgraphs:
        mlp = WrapModuleForSubgraph(mlp)
        state_dict = {f"prefix.{k}": v for k, v in state_dict.items()}

    mlp.load_state_dict(state_dict)

    session = InferenceSession(devices=devices)
    signals = Signals(devices=devices_refs)
    with Graph(
        "MLP",
        input_types=[
            TensorType(
                dtype,
                (
                    x.shape[0],
                    x.shape[1],
                ),
                device=DeviceRef.GPU() if n_gpus > 0 else DeviceRef.CPU(),
            ),
            *signals.input_types(),
        ],
    ) as graph:
        graph_input, *graph_signal_buffers = graph.inputs
        assert isinstance(graph_input, TensorValue)
        assert are_all_buffer_values_sequence(graph_signal_buffers)
        graph_output: Value | list[Value] | list[TensorValue]  # type: ignore[type-arg]
        if n_gpus <= 1:
            graph_output = mlp(graph_input)
        else:
            distributed_inputs = _distribute_value(graph_input, devices)
            mlp_outputs = [
                mlp_shard(x)
                for mlp_shard, x in zip(
                    mlp_shards, distributed_inputs, strict=True
                )
            ]

            graph_output = mlp_allreduce(mlp_outputs, graph_signal_buffers)

        if isinstance(graph_output, list):
            graph.output(*graph_output)
        else:
            graph.output(graph_output)

    compiled = session.load(graph, weights_registry=mlp.state_dict())

    signal_buffers = [
        Buffer.zeros(shape=(Signals.NUM_BYTES,), dtype=DType.uint8, device=dev)
        for dev in devices
    ]

    returned = compiled.execute(x, *signal_buffers)
    returned_tensors = []
    for tensor in returned:
        assert isinstance(tensor, Buffer)
        returned_tensors.append(tensor)
    return returned_tensors


def compare_mlp_outputs(
    hidden_dim: int,
    dim: int,
    activation_function: str,
    torch_dtype: torch.dtype,
    dtype: DType,
    n_gpus: int = 0,
    has_bias: bool = False,
    use_subgraphs: bool = True,
    seq_len: int = 1,
    swiglu_limit: float = 0.0,
) -> None:
    if n_gpus > accelerator_count():
        pytest.skip(f"Not enough GPUs to run test with {n_gpus} GPUs.")

    gate_proj_w = generate_tensor(
        (hidden_dim, dim), torch_dtype, hidden_dim, seed=42
    )
    down_proj_w = generate_tensor(
        (dim, hidden_dim), torch_dtype, hidden_dim, seed=43
    )
    up_proj_w = generate_tensor(
        (hidden_dim, dim), torch_dtype, hidden_dim, seed=44
    )
    if has_bias:
        gate_proj_b = generate_tensor(
            (hidden_dim,), torch_dtype, hidden_dim, seed=42
        )
        down_proj_b = generate_tensor((dim,), torch_dtype, hidden_dim, seed=43)
        up_proj_b = generate_tensor(
            (hidden_dim,), torch_dtype, hidden_dim, seed=44
        )

    device = "cuda" if n_gpus > 0 else "cpu"
    x = generate_tensor((seq_len, dim), torch_dtype, hidden_dim, seed=45).to(
        device
    )

    if has_bias:
        max_output = mlp_output(
            gate_proj_w,
            down_proj_w,
            up_proj_w,
            x,
            activation_function,
            dtype,
            n_gpus=n_gpus,
            has_bias=has_bias,
            bias=(gate_proj_b, down_proj_b, up_proj_b),
            use_subgraphs=use_subgraphs,
            swiglu_limit=swiglu_limit,
        )
    else:
        max_output = mlp_output(
            gate_proj_w,
            down_proj_w,
            up_proj_w,
            x,
            activation_function,
            dtype,
            n_gpus=n_gpus,
            use_subgraphs=use_subgraphs,
            swiglu_limit=swiglu_limit,
        )

    if has_bias:
        torch_output = (
            TorchMLP(
                gate_proj_w.to(device),
                down_proj_w.to(device),
                up_proj_w.to(device),
                activation_function,
                bias=has_bias,
                bias_tensors=(
                    gate_proj_b.to(device),
                    down_proj_b.to(device),
                    up_proj_b.to(device),
                ),
                swiglu_limit=swiglu_limit,
            )(x)
            .detach()
            .to(torch_dtype)
            .to(device)
        )
    else:
        torch_output = (
            TorchMLP(
                gate_proj_w.to(device),
                down_proj_w.to(device),
                up_proj_w.to(device),
                activation_function,
                bias=has_bias,
                swiglu_limit=swiglu_limit,
            )(x)
            .detach()
            .to(torch_dtype)
            .to(device)
        )

    # For the distributed case we need to check all outputs.
    for max_out in max_output:
        torch.testing.assert_close(
            torch_output.to("cpu"),
            torch.from_dlpack(max_out).to(torch_dtype).to("cpu"),
            rtol=2e-1,
            atol=3 * torch.finfo(torch_dtype).eps,
        )
