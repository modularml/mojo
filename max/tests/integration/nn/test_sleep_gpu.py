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

import time

import pytest
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph
from max.nn import kernels


def build_sleep_graph(device: Device) -> Model:
    with Graph(
        "my_sleep_graph",
        input_types=[BufferType(DType.float64, [1], device=DeviceRef.CPU())],
    ) as graph:
        buffer = graph.inputs[0].buffer
        kernels.sleep(buffer, device_ref=DeviceRef.from_device(device))
        graph.output()
    session = InferenceSession(devices=[device])
    model = session.load(graph)
    return model


def time_sleep_kernel_with_buffer(
    seconds_buffer: Buffer,
    device: Device = Accelerator(),  # noqa: B008
) -> float:
    model = build_sleep_graph(device=device)

    t0 = time.time()
    model.execute(seconds_buffer)
    device.synchronize()
    t1 = time.time()
    actual_seconds = t1 - t0

    return actual_seconds


def time_sleep_kernel(seconds: float, device: Device = Accelerator()) -> float:  # noqa: B008
    seconds_buffer = Buffer(shape=[1], dtype=DType.float64)
    seconds_buffer[0] = seconds
    return time_sleep_kernel_with_buffer(seconds_buffer, device)


@pytest.mark.parametrize("device", [Accelerator(), CPU()])
def test_basic(device: Device) -> None:
    desired_seconds = 0.1
    actual_seconds = time_sleep_kernel(seconds=desired_seconds, device=device)

    # Verify the sleep ran for at least the requested duration, with up to
    # 1x overhead for CI load. The exact upper bound is not critical — the
    # test checks that the kernel sleeps at all and doesn't return immediately.
    assert desired_seconds <= actual_seconds <= 2 * desired_seconds


@pytest.mark.parametrize("device", [Accelerator(), CPU()])
def test_zero_sec(device: Device) -> None:
    actual_seconds = time_sleep_kernel(seconds=0.0, device=device)

    # A zero second sleep should complete very quickly.
    assert actual_seconds <= 0.5


@pytest.mark.parametrize("device", [Accelerator(), CPU()])
def test_negative_sec(device: Device) -> None:
    with pytest.raises(
        ValueError, match=r"Sleep duration must be non-negative\. Found: -1\.0"
    ):
        time_sleep_kernel(seconds=-1.0, device=device)


def test_wrong_dtype_graph_building() -> None:
    with pytest.raises(
        ValueError,
        match=r"Expected duration_sec to have DType\.float64 but got DType\.int32",
    ):
        with Graph(
            "my_sleep_graph",
            input_types=[BufferType(DType.int32, [1], device=DeviceRef.CPU())],
        ) as graph:
            buffer = graph.inputs[0].buffer
            kernels.sleep(buffer, device_ref=DeviceRef.GPU())
            graph.output()


def test_wrong_rank_graph_building() -> None:
    with pytest.raises(
        ValueError,
        match=r"Expected duration_sec to have shape \[1\] but got \[\]",
    ):
        with Graph(
            "my_sleep_graph",
            input_types=[BufferType(DType.float64, [], device=DeviceRef.CPU())],
        ) as graph:
            buffer = graph.inputs[0].buffer
            kernels.sleep(buffer, device_ref=DeviceRef.GPU())
            graph.output()


def test_wrong_device_graph_building() -> None:
    with pytest.raises(
        ValueError,
        match=r"Expected duration_sec to be on cpu but got gpu:0",
    ):
        with Graph(
            "my_sleep_graph",
            input_types=[
                BufferType(DType.float64, [1], device=DeviceRef.GPU())
            ],
        ) as graph:
            buffer = graph.inputs[0].buffer
            kernels.sleep(buffer, device_ref=DeviceRef.GPU())
            graph.output()


def test_wrong_dtype_runtime() -> None:
    with pytest.raises(
        ValueError,
        match=r"Input at position 0: Buffer of type \[\(1\), si32\] does not match expected type \[\(1\), f64\]",
    ):
        seconds_buffer = Buffer(shape=[1], dtype=DType.int32)
        time_sleep_kernel_with_buffer(seconds_buffer)


def test_wrong_rank_runtime() -> None:
    with pytest.raises(
        ValueError,
        match=r"Input at position 0: Buffer of type \[\(1, 2, 3\), f64\] does not match expected type \[\(1\), f64\]",
    ):
        seconds_buffer = Buffer(shape=[1, 2, 3], dtype=DType.float64)
        time_sleep_kernel_with_buffer(seconds_buffer)

    with pytest.raises(
        ValueError,
        match=r"Input at position 0: Buffer of type \[\(\), f64\] does not match expected type \[\(1\), f64\]",
    ):
        seconds_buffer = Buffer(shape=[], dtype=DType.float64)
        time_sleep_kernel_with_buffer(seconds_buffer)
