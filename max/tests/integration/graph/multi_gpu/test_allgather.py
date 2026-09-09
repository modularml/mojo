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

"""Test the max.engine Python bindings with Max Graph when using explicit device."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Type, ops
from max.nn import Signals

M = 512
N = 1024


def allgather_graph(signals: Signals, axis: int) -> Graph:
    devices = signals.devices
    with Graph(
        "allgather",
        # https://github.com/python/mypy/issues/19413
        input_types=cast(
            list[Type[Any]],
            [
                TensorType(dtype=DType.float32, shape=[M, N], device=device)
                for device in devices
            ]
            + signals.input_types(),
        ),
    ) as graph:
        num_devices = len(devices)
        for input in graph.inputs[:num_devices]:
            assert isinstance(input, TensorValue)
        allgather_outputs = ops.allgather(
            inputs=(v.tensor for v in graph.inputs[:num_devices]),
            signal_buffers=(v.buffer for v in graph.inputs[num_devices:]),
            axis=axis,
        )
        graph.output(*allgather_outputs)
        return graph


@pytest.mark.parametrize("num_gpus, axis", [(1, 0), (2, 0), (4, 0)])
def test_allgather_execution_even(num_gpus: int, axis: int) -> None:
    """Tests multi-device allgather execution with equal shapes."""

    if num_gpus > accelerator_count():
        pytest.skip(
            f"Not enough GPUs to run allgather test with {num_gpus} GPUs."
        )
    signals = Signals(devices=[DeviceRef.GPU(id=id) for id in range(num_gpus)])
    graph = allgather_graph(signals, axis)
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]

    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    # Set up input values so that the gathered output is a range from
    # 0 to the total number of elements in all inputs.
    numpy_inputs = []
    tensor_inputs = []
    stride = M * N
    numpy_inputs = [
        np.arange(stride).reshape((M, N)) + (stride * i)
        for i in range(num_gpus)
    ]
    tensor_inputs = [
        Buffer.from_numpy(a.astype(np.float32)).to(device)
        for a, device in zip(numpy_inputs, devices, strict=True)
    ]

    outputs = compiled.execute(*tensor_inputs, *signals.buffers())

    expected_output = np.concatenate(numpy_inputs, axis=axis)

    for n, output in enumerate(outputs):
        assert isinstance(output, Buffer)
        assert output.device == devices[n]
        assert np.array_equal(output.to(host).to_numpy(), expected_output)


@pytest.mark.parametrize(
    "shapes, axis",
    [
        (
            [(512, 1024), (512, 1024), (511, 1024), (511, 1024)],
            0,
        ),  # Uneven first dim
        ([(1025, 512), (1024, 512)], 0),  # 2 GPUs with uneven shapes
        (
            [(512, 256), (512, 256), (512, 255), (512, 255)],
            1,
        ),  # Uneven second dim
    ],
)
def test_allgather_execution_uneven(
    shapes: list[tuple[int, int]], axis: int
) -> None:
    """Tests multi-device allgather execution with uneven shapes."""

    num_gpus = len(shapes)
    if num_gpus > accelerator_count():
        pytest.skip(
            f"Not enough GPUs to run allgather test with {num_gpus} GPUs."
        )

    # Create graph with different input shapes
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    signals = Signals(devices)
    with Graph(
        "allgather_uneven",
        # https://github.com/python/mypy/issues/19413
        input_types=cast(
            list[Type[Any]],
            [
                TensorType(dtype=DType.float32, shape=shape, device=device)
                for shape, device in zip(shapes, devices, strict=True)
            ]
            + signals.input_types(),
        ),
    ) as graph:
        allgather_outputs = ops.allgather(
            inputs=(v.tensor for v in graph.inputs[:num_gpus]),
            signal_buffers=(v.buffer for v in graph.inputs[num_gpus:]),
            axis=axis,
        )
        graph.output(*allgather_outputs)

    host = CPU()
    accel_devices = [Accelerator(n) for n in range(num_gpus)]

    session = InferenceSession(devices=[host, *accel_devices])
    compiled = session.load(graph)

    # Set up input values
    numpy_inputs = []
    tensor_inputs = []
    offset = 0
    for i, shape in enumerate(shapes):
        size = int(np.prod(shape))
        arr = np.arange(size).reshape(shape) + offset
        numpy_inputs.append(arr.astype(np.float32))
        tensor_inputs.append(
            Buffer.from_numpy(arr.astype(np.float32)).to(accel_devices[i])
        )
        offset += size

    outputs = compiled.execute(*tensor_inputs, *signals.buffers())

    expected_output = np.concatenate(numpy_inputs, axis=axis)

    for n, output in enumerate(outputs):
        assert isinstance(output, Buffer)
        assert output.device == accel_devices[n]
        assert np.equal(output.to(host).to_numpy(), expected_output).all()


def test_grouped_allgather_execution() -> None:
    """Tests grouped allgather with independent contiguous groups."""
    num_gpus = 4
    group_size = 2
    if num_gpus > accelerator_count():
        pytest.skip(
            f"Not enough GPUs to run allgather test with {num_gpus} GPUs."
        )

    H = 256
    shapes = [(5, H), (4, H), (3, H), (2, H)]
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    signals = Signals(devices)
    with Graph(
        "grouped_allgather",
        input_types=cast(
            list[Type[Any]],
            [
                TensorType(dtype=DType.float32, shape=shape, device=device)
                for shape, device in zip(shapes, devices, strict=True)
            ]
            + signals.input_types(),
        ),
    ) as graph:
        allgather_outputs = ops.allgather(
            inputs=(v.tensor for v in graph.inputs[:num_gpus]),
            signal_buffers=(v.buffer for v in graph.inputs[num_gpus:]),
            axis=0,
            group_size=group_size,
        )
        graph.output(*allgather_outputs)

    host = CPU()
    accel_devices = [Accelerator(n) for n in range(num_gpus)]
    session = InferenceSession(devices=[host, *accel_devices])
    compiled = session.load(graph)

    numpy_inputs = []
    tensor_inputs = []
    offset = 0
    for shape, device in zip(shapes, accel_devices, strict=True):
        size = int(np.prod(shape))
        arr = np.arange(size, dtype=np.float32).reshape(shape) + offset
        numpy_inputs.append(arr)
        tensor_inputs.append(Buffer.from_numpy(arr).to(device))
        offset += size

    outputs = compiled.execute(*tensor_inputs, *signals.buffers())

    expected_by_group = []
    for group_start in range(0, num_gpus, group_size):
        expected_by_group.append(
            np.concatenate(
                numpy_inputs[group_start : group_start + group_size], axis=0
            )
        )

    for dev_idx, output in enumerate(outputs):
        assert isinstance(output, Buffer)
        assert output.device == accel_devices[dev_idx]
        expected = expected_by_group[dev_idx // group_size]
        assert np.array_equal(output.to(host).to_numpy(), expected)
