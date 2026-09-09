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
"""Test device event creation, recording, and synchronization functionality."""

from typing import Any

import pytest
from max import driver
from max.driver import DevicePinnedBuffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.fixture
def device() -> driver.Accelerator:
    """Fixture to provide an accelerator device."""
    return driver.Accelerator()


@pytest.fixture
def simple_graph(device: driver.Accelerator) -> Graph:
    """Fixture to create a simple computation graph for testing."""
    width = 1024 * 1024
    device_ref = DeviceRef.from_device(device)
    with Graph(
        "add_graph",
        input_types=[
            TensorType(DType.int32, [width], device=device_ref),
            TensorType(DType.int32, [width], device=device_ref),
        ],
    ) as graph:
        a, b = graph.inputs
        result = ops.add(a, b)
        graph.output(result)
    return graph


@pytest.fixture
def model(device: driver.Accelerator, simple_graph: Graph) -> Any:
    """Fixture to create a loaded model for testing."""
    session = InferenceSession(devices=[device])
    return session.load(simple_graph)


class TestEventRecording:
    """Test event recording on streams."""

    def test_record_and_synchronize(
        self, device: driver.Accelerator, model: Any
    ) -> None:
        """Test recording events and synchronization."""
        width = 1024 * 1024
        a_buf = DevicePinnedBuffer(DType.int32, [width], device=device)
        a_buf.to_numpy().fill(1)
        b_buf = DevicePinnedBuffer(DType.int32, [width], device=device)
        b_buf.to_numpy().fill(2)

        a_dev = a_buf.to(device)
        b_dev = b_buf.to(device)

        # Test both record_event() overloads
        event1 = device.default_queue.record_event()  # Creates new event
        model.execute(a_dev, b_dev)

        event2 = driver.DeviceEvent(device)
        device.default_queue.record_event(event2)  # Records existing event

        # Synchronize and verify
        event2.synchronize()
        assert event1.is_ready()
        assert event2.is_ready()

    def test_elapsed_time_with_compute(
        self, device: driver.Accelerator, model: Any
    ) -> None:
        """Test elapsed_time around actual GPU compute returns positive time."""
        width = 1024 * 1024
        a_buf = DevicePinnedBuffer(DType.int32, [width], device=device)
        a_buf.to_numpy().fill(1)
        b_buf = DevicePinnedBuffer(DType.int32, [width], device=device)
        b_buf.to_numpy().fill(2)

        a_dev = a_buf.to(device)
        b_dev = b_buf.to(device)

        start = driver.DeviceEvent(device, enable_timing=True)
        end = driver.DeviceEvent(device, enable_timing=True)

        stream = device.default_queue
        stream.record_event(start)
        model.execute(a_dev, b_dev)
        stream.record_event(end)
        end.synchronize()

        elapsed = start.elapsed_time(end)
        assert isinstance(elapsed, float)
        assert elapsed > 0.0, f"Expected positive elapsed time, got {elapsed}"
