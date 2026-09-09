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

"""Test the max.engine Python bindings with reducescatter operation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    Device,
    accelerator_count,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn import Module, Signals
from max.support.math import ceildiv

M = 512
N = 1024


def reducescatter_graph(signals: Signals) -> Graph:
    devices = signals.devices
    num_devices = len(devices)

    # Create input types for each device
    input_types = [
        TensorType(dtype=DType.float32, shape=[M, N], device=devices[i])
        for i in range(num_devices)
    ]
    # Combine tensor types and buffer types
    all_input_types = input_types + list(signals.input_types())

    with Graph(
        "reducescatter",
        input_types=all_input_types,
    ) as graph:
        # Get tensor inputs and apply scaling
        tensor_inputs = []
        for i in range(num_devices):
            assert isinstance(graph.inputs[i], TensorValue)
            # Scale each input by (i + 1)
            scaled_input = graph.inputs[i].tensor * (i + 1)
            tensor_inputs.append(scaled_input)

        reducescatter_outputs = ops.reducescatter.sum(
            tensor_inputs,
            [inp.buffer for inp in graph.inputs[num_devices:]],
        )

        graph.output(*reducescatter_outputs)
        return graph


def test_reducescatter_execution() -> None:
    """Tests multi-device reducescatter execution."""
    # Use available GPUs, minimum 2, maximum 4
    available_gpus = accelerator_count()
    if available_gpus < 2:
        pytest.skip("Test requires at least 2 GPUs")

    num_gpus = min(available_gpus, 4)

    signals = Signals(devices=[DeviceRef.GPU(id=id) for id in range(num_gpus)])
    graph = reducescatter_graph(signals)
    host = CPU()

    # Create device objects
    devices: list[Device]
    devices = [Accelerator(i) for i in range(num_gpus)]

    session = InferenceSession(devices=[host] + devices)
    compiled = session.load(graph)

    # Create input tensors
    a_np = np.ones((M, N)).astype(np.float32)
    # Expected reduced value: sum of (1 * 1) + (1 * 2) + ... + (1 * num_gpus)
    # = 1 + 2 + ... + num_gpus = num_gpus * (num_gpus + 1) / 2
    expected_sum = num_gpus * (num_gpus + 1) // 2

    # Create tensors on each device
    input_tensors = [Buffer.from_numpy(a_np).to(device) for device in devices]

    output = compiled.execute(*input_tensors, *signals.buffers())

    # Check Executed Graph
    # Output shape should be [M, N/num_gpus] for each device
    expected_rows = M // num_gpus
    for out_tensor, device in zip(output, devices, strict=True):
        assert isinstance(out_tensor, Buffer)
        assert out_tensor.device == device
        result = out_tensor.to(host).to_numpy()
        assert result.shape == (expected_rows, N)
        # Each device gets a portion of the reduced result
        expected_out = np.full(
            (expected_rows, N), expected_sum, dtype=np.float32
        )
        assert np.allclose(expected_out, result)


class ReduceScatterAdd(Module):
    """A fused reducescatter with an elementwise add."""

    num_devices: int
    """Number of devices to reducescatter between."""

    def __init__(self, num_devices: int) -> None:
        super().__init__()
        self.num_devices = num_devices

    def __call__(
        self,
        *args: TensorValue | BufferValue,
    ) -> list[TensorValue]:
        # Split args into tensor inputs and signal buffers
        inputs = [cast(TensorValue, arg) for arg in args[: self.num_devices]]
        signal_buffers = [
            cast(BufferValue, arg) for arg in args[self.num_devices :]
        ]

        # Reducescatter
        results = ops.reducescatter.sum(inputs, signal_buffers)

        biases = [
            ops.constant(42, dtype=DType.float32, device=DeviceRef.GPU(id))
            for id in range(self.num_devices)
        ]

        # Elementwise add that should fuse into reducescatter's epilogue.
        return [x + y for x, y in zip(results, biases, strict=True)]


def _reducescatter_axis_graph(
    signals: Signals, axis: int, shape: tuple[int, int]
) -> Graph:
    """Build a reducescatter graph that scatters along the given axis."""
    devices = signals.devices
    num_devices = len(devices)

    input_types = [
        TensorType(dtype=DType.float32, shape=list(shape), device=devices[i])
        for i in range(num_devices)
    ]
    all_input_types = input_types + list(signals.input_types())

    with Graph(
        "reducescatter_axis",
        input_types=all_input_types,
    ) as graph:
        tensor_inputs = [graph.inputs[i].tensor for i in range(num_devices)]
        reducescatter_outputs = ops.reducescatter.sum(
            tensor_inputs,
            [inp.buffer for inp in graph.inputs[num_devices:]],
            axis=axis,
        )
        graph.output(*reducescatter_outputs)
        return graph


def _grouped_reducescatter_axis0_graph(
    signals: Signals, shapes: list[tuple[int, int]], group_size: int
) -> Graph:
    """Build a grouped reducescatter graph that scatters rows."""
    devices = signals.devices
    num_devices = len(devices)

    input_types = [
        TensorType(dtype=DType.float32, shape=list(shape), device=devices[i])
        for i, shape in enumerate(shapes)
    ]
    all_input_types = input_types + list(signals.input_types())

    with Graph(
        "grouped_reducescatter_axis0",
        input_types=all_input_types,
    ) as graph:
        tensor_inputs = [graph.inputs[i].tensor for i in range(num_devices)]
        reducescatter_outputs = ops.reducescatter.sum(
            tensor_inputs,
            [inp.buffer for inp in graph.inputs[num_devices:]],
            axis=0,
            group_size=group_size,
        )
        graph.output(*reducescatter_outputs)
        return graph


@pytest.mark.parametrize("num_gpus", [2, 4, 8])
def test_reducescatter_axis0_execution(num_gpus: int) -> None:
    """Tests reducescatter with scatter on axis 0 (rows)."""
    if (available_gpus := accelerator_count()) < num_gpus:
        pytest.skip(
            f"skipping {num_gpus=} test since only {available_gpus} available"
        )

    shape = (M, N)
    graph_devices = [DeviceRef.GPU(id) for id in range(num_gpus)]
    signals = Signals(devices=graph_devices)
    graph = _reducescatter_axis_graph(signals, axis=0, shape=shape)

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host] + devices)
    compiled = session.load(graph)

    # Use row-varying data so incorrect axis would produce wrong values.
    # Each device sends the same data: row i has value (i+1).
    base = np.repeat(
        np.arange(1, M + 1, dtype=np.float32).reshape(M, 1), N, axis=1
    )
    input_tensors = [Buffer.from_numpy(base).to(dev) for dev in devices]

    outputs = compiled.execute(*input_tensors, *signals.buffers())

    # After reduction (sum of num_gpus identical inputs), values = base * num_gpus.
    # Scatter axis=0: rows are partitioned across devices.
    expected_rows = M // num_gpus
    reduced = base * num_gpus
    for dev_idx, (out_tensor, device) in enumerate(
        zip(outputs, devices, strict=True)
    ):
        assert isinstance(out_tensor, Buffer)
        assert out_tensor.device == device
        result = out_tensor.to(host).to_numpy()
        assert result.shape == (expected_rows, N)
        row_start = dev_idx * expected_rows
        expected = reduced[row_start : row_start + expected_rows, :]
        assert np.allclose(expected, result)


@pytest.mark.parametrize("num_gpus", [2, 4, 8])
def test_reducescatter_axis1_execution(num_gpus: int) -> None:
    """Tests reducescatter with scatter on axis 1 (columns)."""
    if (available_gpus := accelerator_count()) < num_gpus:
        pytest.skip(
            f"skipping {num_gpus=} test since only {available_gpus} available"
        )

    shape = (M, N)
    graph_devices = [DeviceRef.GPU(id) for id in range(num_gpus)]
    signals = Signals(devices=graph_devices)
    graph = _reducescatter_axis_graph(signals, axis=1, shape=shape)

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host] + devices)
    compiled = session.load(graph)

    # Use column-varying data so incorrect axis would produce wrong values.
    # Each device sends the same data: column j has value (j+1).
    base = np.repeat(
        np.arange(1, N + 1, dtype=np.float32).reshape(1, N), M, axis=0
    )
    input_tensors = [Buffer.from_numpy(base).to(dev) for dev in devices]

    outputs = compiled.execute(*input_tensors, *signals.buffers())

    # After reduction, values = base * num_gpus.
    # Scatter axis=1: columns are partitioned across devices.
    expected_cols = N // num_gpus
    reduced = base * num_gpus
    for dev_idx, (out_tensor, device) in enumerate(
        zip(outputs, devices, strict=True)
    ):
        assert isinstance(out_tensor, Buffer)
        assert out_tensor.device == device
        result = out_tensor.to(host).to_numpy()
        assert result.shape == (M, expected_cols)
        col_start = dev_idx * expected_cols
        expected = reduced[:, col_start : col_start + expected_cols]
        assert np.allclose(expected, result)


def test_grouped_reducescatter_axis0_execution() -> None:
    """Tests grouped reduce-scatter with different per-group row counts."""
    num_gpus = 4
    group_size = 2
    if (available_gpus := accelerator_count()) < num_gpus:
        pytest.skip(
            f"skipping {num_gpus=} test since only {available_gpus} available"
        )

    H = 256
    shapes = [(5, H), (5, H), (3, H), (3, H)]
    graph_devices = [DeviceRef.GPU(id) for id in range(num_gpus)]
    signals = Signals(devices=graph_devices)
    graph = _grouped_reducescatter_axis0_graph(
        signals, shapes=shapes, group_size=group_size
    )

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host] + devices)
    compiled = session.load(graph)

    inputs = []
    reduced_by_group: list[np.ndarray] = []
    for group_start in range(0, num_gpus, group_size):
        group_rows = shapes[group_start][0]
        base = np.repeat(
            np.arange(1, group_rows + 1, dtype=np.float32).reshape(
                group_rows, 1
            ),
            H,
            axis=1,
        )
        group_inputs = []
        for local_idx in range(group_size):
            dev_idx = group_start + local_idx
            group_inputs.append(base * (dev_idx + 1))
            inputs.append(
                Buffer.from_numpy(group_inputs[-1]).to(devices[dev_idx])
            )
        reduced_by_group.append(np.sum(group_inputs, axis=0))

    outputs = compiled.execute(*inputs, *signals.buffers())

    for group_idx, group_start in enumerate(range(0, num_gpus, group_size)):
        group_rows = shapes[group_start][0]
        chunk_sizes = [
            (group_rows + (group_size - local_idx - 1)) // group_size
            for local_idx in range(group_size)
        ]
        row = 0
        for local_idx, rows in enumerate(chunk_sizes):
            dev_idx = group_start + local_idx
            out_tensor = outputs[dev_idx]
            assert isinstance(out_tensor, Buffer)
            assert out_tensor.device == devices[dev_idx]
            result = out_tensor.to(host).to_numpy()
            assert result.shape == (rows, H)
            expected = reduced_by_group[group_idx][row : row + rows]
            assert np.allclose(expected, result)
            row += rows


@pytest.mark.parametrize("num_gpus", [2, 4, 8])
def test_reducescatter_epilogue_fusion(num_gpus: int) -> None:
    """Tests that an elementwise add correctly follows a reducescatter operation."""
    if (available_gpus := accelerator_count()) < num_gpus:
        pytest.skip(
            f"skipping {num_gpus=} test since only {available_gpus} available"
        )

    graph_devices = [DeviceRef.GPU(id) for id in range(num_gpus)]
    signals = Signals(devices=graph_devices)

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host] + devices)

    model = ReduceScatterAdd(num_devices=len(devices))
    graph = Graph(
        "ReduceScatterAdd_fusion",
        forward=model,
        input_types=[
            *[
                TensorType(DType.float32, shape=[M, N], device=graph_devices[i])
                for i in range(num_gpus)
            ],
            *signals.input_types(),
        ],
    )

    compiled = session.load(graph)

    inputs = []
    a_np = np.ones((M, N), np.float32)
    for i in range(num_gpus):
        inputs.append(Buffer.from_numpy(a_np).to(devices[i]))

    for dev in devices:
        dev.synchronize()

    outputs = compiled.execute(*inputs, *signals.buffers())

    # Expected: sum of all inputs (num_gpus ones) + 42 bias
    # Each input is ones, so sum = num_gpus, plus bias = 42
    expected_rows = M // num_gpus
    expected = np.full((expected_rows, N), num_gpus + 42.0, dtype=np.float32)

    for tensor in outputs:
        assert isinstance(tensor, Buffer)
        result = tensor.to(host).to_numpy()
        assert result.shape == (expected_rows, N)
        assert np.allclose(expected, result, atol=1e-6)


def test_reducescatter_symbolic_dim(
    symbolic_reducescatter_model: Model | None, num_gpus: int
) -> None:
    """Tests reducescatter with a symbolic dimension on the scatter axis."""
    if symbolic_reducescatter_model is None:
        pytest.skip("not enough GPUs available")

    H = 256
    scatter_size = 512

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    signals = Signals(devices=[DeviceRef.GPU(id) for id in range(num_gpus)])

    base = np.ones((scatter_size, H), dtype=np.float32)
    input_tensors = [Buffer.from_numpy(base).to(dev) for dev in devices]
    outputs = symbolic_reducescatter_model.execute(
        *input_tensors, *signals.buffers()
    )

    expected_shape = (scatter_size // num_gpus, H)
    expected = np.full(expected_shape, num_gpus, dtype=np.float32)

    for out_tensor, device in zip(outputs, devices, strict=True):
        assert isinstance(out_tensor, Buffer)
        assert out_tensor.device == device
        result = out_tensor.to(host).to_numpy()
        assert result.shape == expected_shape
        assert np.allclose(expected, result)


def test_reducescatter_symbolic_ragged(
    symbolic_reducescatter_model: Model | None, num_gpus: int
) -> None:
    """Tests reducescatter with symbolic dim and uneven (ragged) split."""
    if symbolic_reducescatter_model is None:
        pytest.skip("not enough GPUs available")

    H = 256
    actual_seq_len = 3

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    signals = Signals(devices=[DeviceRef.GPU(id) for id in range(num_gpus)])

    base = np.arange(1, actual_seq_len * H + 1, dtype=np.float32).reshape(
        actual_seq_len, H
    )
    input_tensors = [Buffer.from_numpy(base).to(dev) for dev in devices]
    result_outputs = symbolic_reducescatter_model.execute(
        *input_tensors, *signals.buffers()
    )

    reduced = base * num_gpus
    row = 0
    for dev_idx, out_tensor in enumerate(result_outputs):
        assert isinstance(out_tensor, Buffer)
        expected_rows = min(
            ceildiv(actual_seq_len, num_gpus), actual_seq_len - row
        )
        result = out_tensor.to(host).to_numpy()
        assert result.shape == (expected_rows, H), (
            f"device {dev_idx}: expected {(expected_rows, H)}, "
            f"got {result.shape}"
        )
        if expected_rows > 0:
            expected = reduced[row : row + expected_rows]
            assert np.allclose(expected, result)
            row += expected_rows
